#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""CP-SAT solvers for capacitated multi-item lot sizing problem."""

import logging
from enum import Enum
from typing import Any

from ortools.sat.python.cp_model import CpSolverSolutionCallback

from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
)
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver
from discrete_optimization.lotsizing import ProductionDecision
from discrete_optimization.lotsizing.capacitatedmultiitem.problem import (
    CapacitatedMultiItemLSP,
    CapacitatedMultiItemSolution,
)

logger = logging.getLogger(__name__)


class ChangeoverModel(Enum):
    """Modeling approach for changeover costs in CP-SAT solver."""

    STATE_BASED = "state_based"
    TRANSITION_BASED = "transition_based"
    SHORTEST_PATH_BASED = "shortest_path_based"


class CpSatLotSizingSolver(OrtoolsCpSatSolver, WarmstartMixin):
    """CP-SAT solver for capacitated multi-item lot sizing.

    Supports multiple changeover cost encodings:
    - STATE_BASED: Track last produced item using element constraints
    - TRANSITION_BASED: Model explicit transitions between production events
    - SHORTEST_PATH_BASED: Model as shortest path through production sequence

    Supports warm-start from existing solutions.
    """

    problem: CapacitatedMultiItemLSP
    variables: dict

    hyperparameters = [
        EnumHyperparameter(
            name="changeover_model",
            enum=ChangeoverModel,
            default=ChangeoverModel.STATE_BASED,
        ),
    ]

    def init_model(self, **kwargs: Any) -> None:
        """Initialize the CP-SAT model.

        Args:
            changeover_model: How to model changeover costs (default: STATE_BASED)
            **kwargs: Additional parameters passed to parent class
        """
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        super().init_model(**kwargs)
        self._create_main_vars()
        self._set_objective(changeover_model=kwargs["changeover_model"])

    def _create_main_vars(self):
        """Create main decision variables and constraints."""
        self.variables = {}
        total_demands_per_item = {
            item: self.problem.get_total_demand(item)
            for item in self.problem.items_list
        }
        horizon = self.problem.horizon

        # Boolean variables: is item produced at time t?
        bool_produce_type_time = {
            (item, t): self.cp_model.NewBoolVar(name=f"bool_prod_item_{item}_time_{t}")
            for item in self.problem.items_list
            for t in range(horizon)
        }

        # At most one item type produced per time period
        for t in range(horizon):
            self.cp_model.add_at_most_one(
                [bool_produce_type_time[(item, t)] for item in self.problem.items_list]
            )

        # Quantity produced variables
        if self.problem.is_binary:
            quantity_produce = bool_produce_type_time
        else:
            quantity_produce = {
                (item, t): self.cp_model.NewIntVar(
                    lb=0,
                    ub=min(
                        int(self.problem.get_available_production_time(t)),
                        total_demands_per_item[item],
                    ),
                    name=f"prod_item_{item}_time_{t}",
                )
                for item in self.problem.items_list
                for t in range(horizon)
            }
            # Link quantity to boolean
            for item, t in quantity_produce:
                self.cp_model.Add(quantity_produce[(item, t)] >= 1).OnlyEnforceIf(
                    bool_produce_type_time[(item, t)]
                )
                self.cp_model.Add(quantity_produce[(item, t)] == 0).OnlyEnforceIf(
                    bool_produce_type_time[(item, t)].Not()
                )

        # Delivery variables
        delivery = {
            (item, t): self.cp_model.NewIntVar(
                lb=0,
                ub=total_demands_per_item[item],
                name=f"delivery_item_{item}_time_{t}",
            )
            for item in self.problem.items_list
            for t in range(horizon)
        }

        # Stock variables
        stocks = {
            (item, t): self.cp_model.NewIntVar(
                lb=0,
                ub=total_demands_per_item[item],
                name=f"stock_item_{item}_time_{t}",
            )
            for item in self.problem.items_list
            for t in range(horizon)
        }

        # Delay/backlog variables
        delays = {
            (item, t): self.cp_model.NewIntVar(
                lb=0,
                ub=0
                if not self.problem.is_backlog_allowed()
                else total_demands_per_item[item],
                name=f"delays_item_{item}_time_{t}",
            )
            for item in self.problem.items_list
            for t in range(horizon)
        }

        # Stock balance constraints
        for item in self.problem.items_list:
            for t in range(horizon):
                prev_stock = 0 if t == 0 else stocks[(item, t - 1)]
                demand = self.problem.get_demand(item, t)

                # stock[t] = stock[t-1] + production[t] - delivery[t]
                self.cp_model.Add(
                    stocks[(item, t)]
                    == prev_stock + quantity_produce[(item, t)] - delivery[(item, t)]
                )

                # Delay tracking: delay[t] = delay[t-1] + demand[t] - delivery[t]
                prev_delay = 0 if t == 0 else delays[(item, t - 1)]
                self.cp_model.Add(
                    delays[(item, t)] == prev_delay + demand - delivery[(item, t)]
                )

        self.variables["deliveries"] = delivery
        self.variables["bool_productions"] = bool_produce_type_time
        self.variables["productions"] = quantity_produce
        self.variables["delays"] = delays
        self.variables["stocks"] = stocks

    def _create_changeover_vars_state_based(self):
        """Create changeover variables using state-based model (element constraints).

        Most efficient: O(horizon) variables.
        """
        produce = self.variables["bool_productions"]
        horizon = self.problem.horizon

        # Which item was produced in each period (nb_items = idle)
        item_produced = {
            t: self.cp_model.NewIntVar(
                lb=0, ub=self.problem.nb_items, name=f"item_at_{t}"
            )
            for t in range(horizon)
        }

        # Link item_produced to bool_productions
        for t in range(horizon):
            for item in self.problem.items_list:
                # If item produced at t, then item_produced[t] = item
                self.cp_model.Add(item_produced[t] == item).OnlyEnforceIf(
                    produce[(item, t)]
                )

        # Changeover cost variables
        changeover_costs_vars = []
        for t in range(1, horizon):
            # Cost from item at t-1 to item at t
            cost_var = self.cp_model.NewIntVar(
                lb=0,
                ub=int(max(max(row) for row in self.problem._changeover_costs)),
                name=f"changeover_cost_{t}",
            )

            # Use AddElement to lookup changeover cost
            # cost = changeover_costs[item_produced[t-1]][item_produced[t]]
            # Flat costs matrix: (nb_items+1) x (nb_items+1) to include idle state
            flat_costs = []
            for i in range(self.problem.nb_items + 1):  # Include idle (nb_items)
                for j in range(self.problem.nb_items + 1):  # Include idle
                    if i == self.problem.nb_items or j == self.problem.nb_items:
                        # Changeover from/to idle has zero cost
                        flat_costs.append(0)
                    else:
                        # Regular item-to-item changeover
                        flat_costs.append(int(self.problem.get_changeover_cost(i, j)))

            # Index = item_produced[t-1] * (nb_items+1) + item_produced[t]
            index = self.cp_model.NewIntVar(
                lb=0,
                ub=(self.problem.nb_items + 1) ** 2 - 1,
                name=f"changeover_index_{t}",
            )
            self.cp_model.Add(
                index
                == item_produced[t - 1] * (self.problem.nb_items + 1) + item_produced[t]
            )

            self.cp_model.AddElement(index, flat_costs, cost_var)
            changeover_costs_vars.append(cost_var)

        self.variables["changeover_costs"] = changeover_costs_vars

    def _create_changeover_vars_transition_based(self):
        """Create changeover variables using transition-based model.

        Explicitly model transitions between production events.
        O(n_items^2 * horizon^2) variables in worst case.
        """
        produce = self.variables["bool_productions"]
        horizon = self.problem.horizon

        # Binary variables: transition from (item0, t) to (item1, t')
        lookahead = min(
            10,
            horizon
            - sum(self.problem.get_total_demand(i) for i in self.problem.items_list),
        )
        lookahead = max(1, lookahead)

        transition = {}
        for item0 in self.problem.items_list:
            for item1 in self.problem.items_list:
                for t in range(horizon):
                    for tprime in range(t + 1, min(t + lookahead + 1, horizon)):
                        transition[(item0, t, item1, tprime)] = (
                            self.cp_model.NewBoolVar(
                                name=f"trans_{item0}_{t}_to_{item1}_{tprime}"
                            )
                        )

        # Transition constraints
        for item0, t, item1, tprime in transition:
            # transition => produce[item0, t] AND produce[item1, tprime]
            self.cp_model.AddImplication(
                transition[(item0, t, item1, tprime)], produce[(item0, t)]
            )
            self.cp_model.AddImplication(
                transition[(item0, t, item1, tprime)], produce[(item1, tprime)]
            )

        # Each production must have outgoing transition (except last)
        for item in self.problem.items_list:
            for t in range(horizon - lookahead):
                outgoing = [
                    transition[(item, t, item1, tprime)]
                    for item1 in self.problem.items_list
                    for tprime in range(t + 1, min(t + lookahead + 1, horizon))
                    if (item, t, item1, tprime) in transition
                ]
                if outgoing:
                    self.cp_model.Add(sum(outgoing) >= 1).OnlyEnforceIf(
                        produce[(item, t)]
                    )

        # Changeover cost
        changeover_cost_terms = []
        for item0, t, item1, tprime in transition:
            cost = int(self.problem.get_changeover_cost(item0, item1))
            if cost > 0:
                changeover_cost_terms.append(
                    cost * transition[(item0, t, item1, tprime)]
                )

        changeover_cost_var = self.cp_model.NewIntVar(
            lb=0,
            ub=sum(
                int(max(self.problem._changeover_costs[i]))
                for i in range(self.problem.nb_items)
            )
            * horizon,
            name="total_changeover_cost",
        )
        if changeover_cost_terms:
            self.cp_model.Add(changeover_cost_var == sum(changeover_cost_terms))
        else:
            self.cp_model.Add(changeover_cost_var == 0)

        self.variables["changeover_cost_var"] = changeover_cost_var

    def _create_changeover_vars_shortest_path(self):
        """Create changeover variables using shortest path model.

        Model production sequence as shortest path through production events.
        Similar to TSP formulation. O(n_productions^2) variables.
        """
        produce = self.variables["bool_productions"]
        horizon = self.problem.horizon

        # Create dummy start and end nodes
        # Transition variables: from (item, t) or "start" to (item', t') or "end"
        lookahead = min(
            10,
            horizon
            - sum(self.problem.get_total_demand(i) for i in self.problem.items_list),
        )
        lookahead = max(1, lookahead)

        transitions = {}

        # Transitions from start to first productions
        for item in self.problem.items_list:
            for t in range(min(lookahead + 1, horizon)):
                transitions[("start", item, t)] = self.cp_model.NewBoolVar(
                    name=f"trans_start_to_{item}_{t}"
                )

        # Transitions between productions
        for item0 in self.problem.items_list:
            for t in range(horizon):
                for item1 in self.problem.items_list:
                    for tprime in range(t + 1, min(t + lookahead + 1, horizon)):
                        transitions[(item0, t, item1, tprime)] = (
                            self.cp_model.NewBoolVar(
                                name=f"trans_{item0}_{t}_to_{item1}_{tprime}"
                            )
                        )

        # Transitions to end
        for item in self.problem.items_list:
            for t in range(max(0, horizon - lookahead - 1), horizon):
                transitions[(item, t, "end")] = self.cp_model.NewBoolVar(
                    name=f"trans_{item}_{t}_to_end"
                )

        # Flow conservation constraints
        # Start: exactly one outgoing
        outgoing_start = [v for k, v in transitions.items() if k[0] == "start"]
        self.cp_model.Add(sum(outgoing_start) == 1)

        # End: exactly one incoming
        incoming_end = [v for k, v in transitions.items() if k[-1] == "end"]
        self.cp_model.Add(sum(incoming_end) == 1)

        # Production nodes: incoming = outgoing = bool_produce
        for item in self.problem.items_list:
            for t in range(horizon):
                node = (item, t)

                # Incoming transitions
                incoming = [
                    v
                    for k, v in transitions.items()
                    if len(k) >= 3 and k[-2] == item and k[-1] == t and k[0] != "start"
                ]
                incoming += [
                    v
                    for k, v in transitions.items()
                    if k[0] == "start" and k[1] == item and k[2] == t
                ]

                # Outgoing transitions
                outgoing = [
                    v
                    for k, v in transitions.items()
                    if len(k) >= 3 and k[0] == item and k[1] == t and k[-1] != "end"
                ]
                outgoing += [
                    v
                    for k, v in transitions.items()
                    if k[0] == item and k[1] == t and k[2] == "end"
                ]

                if incoming:
                    self.cp_model.Add(sum(incoming) == produce[node])
                if outgoing:
                    self.cp_model.Add(sum(outgoing) == produce[node])

        # Changeover cost
        changeover_cost_terms = []
        for key, var in transitions.items():
            if key[0] == "start" or key[-1] == "end":
                continue  # No cost for start/end
            item0, t0, item1, t1 = key
            cost = int(self.problem.get_changeover_cost(item0, item1))
            if cost > 0:
                changeover_cost_terms.append(cost * var)

        changeover_cost_var = self.cp_model.NewIntVar(
            lb=0,
            ub=sum(
                int(max(self.problem._changeover_costs[i]))
                for i in range(self.problem.nb_items)
            )
            * horizon,
            name="total_changeover_cost",
        )
        if changeover_cost_terms:
            self.cp_model.Add(changeover_cost_var == sum(changeover_cost_terms))
        else:
            self.cp_model.Add(changeover_cost_var == 0)

        self.variables["changeover_cost_var"] = changeover_cost_var

    def _set_objective(self, changeover_model: ChangeoverModel):
        """Set the objective function."""
        horizon = self.problem.horizon
        delays = self.variables["delays"]
        stocks = self.variables["stocks"]
        objectives = []

        # Delay cost
        delay_cost_terms = []
        for t in range(horizon):
            for item in self.problem.items_list:
                cost_per_unit = int(self.problem.get_backlog_cost_per_unit(item, t))
                if cost_per_unit > 0:
                    delay_cost_terms.append(cost_per_unit * delays[(item, t)])

        delay_cost_var = None
        if delay_cost_terms:
            delay_cost_var = self.cp_model.NewIntVar(
                lb=0,
                ub=sum(
                    int(self.problem.get_backlog_cost_per_unit(i, t))
                    * self.problem.get_total_demand(i)
                    for i in self.problem.items_list
                    for t in range(horizon)
                ),
                name="delay_cost",
            )
            self.cp_model.Add(delay_cost_var == sum(delay_cost_terms))
            objectives.append(delay_cost_var)

        # Stock cost
        stock_cost_terms = []
        for t in range(horizon):
            for item in self.problem.items_list:
                cost_per_unit = int(self.problem.get_inventory_cost_per_unit(item, t))
                if cost_per_unit > 0:
                    stock_cost_terms.append(cost_per_unit * stocks[(item, t)])

        stock_cost_var = None
        if stock_cost_terms:
            stock_cost_var = self.cp_model.NewIntVar(
                lb=0,
                ub=sum(
                    int(self.problem.get_inventory_cost_per_unit(i, t))
                    * self.problem.get_total_demand(i)
                    * horizon
                    for i in self.problem.items_list
                    for t in range(horizon)
                ),
                name="stock_cost",
            )
            self.cp_model.Add(stock_cost_var == sum(stock_cost_terms))
            objectives.append(stock_cost_var)

        # Store objective component variables for logging
        self.variables["delay_cost_var"] = delay_cost_var
        self.variables["stock_cost_var"] = stock_cost_var

        # Changeover cost
        if changeover_model == ChangeoverModel.STATE_BASED:
            self._create_changeover_vars_state_based()
            if "changeover_costs" in self.variables:
                objectives.extend(self.variables["changeover_costs"])
        elif changeover_model == ChangeoverModel.TRANSITION_BASED:
            self._create_changeover_vars_transition_based()
            if "changeover_cost_var" in self.variables:
                objectives.append(self.variables["changeover_cost_var"])
        elif changeover_model == ChangeoverModel.SHORTEST_PATH_BASED:
            self._create_changeover_vars_shortest_path()
            if "changeover_cost_var" in self.variables:
                objectives.append(self.variables["changeover_cost_var"])

        # Minimize total cost
        if objectives:
            self.cp_model.Minimize(sum(objectives))

    def set_warm_start(self, solution: CapacitatedMultiItemSolution) -> None:
        """Set warm-start hints from a solution.

        Args:
            solution: A solution to use as warm-start
        """
        self.cp_model.ClearHints()

        # Set production hints
        for item in self.problem.items_list:
            for t in range(self.problem.horizon):
                qty = solution.get_production_quantity(item, t)
                if qty > 0:
                    # Hint that this item is produced at this time
                    self.cp_model.AddHint(
                        self.variables["bool_productions"][(item, t)], 1
                    )
                    if not self.problem.is_binary:
                        self.cp_model.AddHint(
                            self.variables["productions"][(item, t)], qty
                        )
                else:
                    # Hint that this item is not produced
                    self.cp_model.AddHint(
                        self.variables["bool_productions"][(item, t)], 0
                    )
                    if not self.problem.is_binary:
                        self.cp_model.AddHint(
                            self.variables["productions"][(item, t)], 0
                        )

        # Set delivery hints
        for item in self.problem.items_list:
            for t in range(self.problem.horizon):
                delivery = solution.get_delivery_quantity(item, t)
                self.cp_model.AddHint(self.variables["deliveries"][(item, t)], delivery)

        # Set stock hints
        for item in self.problem.items_list:
            for t in range(self.problem.horizon):
                stock = solution.get_inventory_level(item, t)
                self.cp_model.AddHint(self.variables["stocks"][(item, t)], stock)

        # Set delay hints
        for item in self.problem.items_list:
            for t in range(self.problem.horizon):
                delay = solution.get_backlog_quantity(item, t)
                self.cp_model.AddHint(self.variables["delays"][(item, t)], delay)

    def retrieve_solution(
        self, cpsolvercb: CpSolverSolutionCallback
    ) -> CapacitatedMultiItemSolution:
        """Extract solution from CP-SAT solver."""
        # Log CP-SAT objective components (from the objective variables)
        cpsat_delay_cost = 0
        cpsat_stock_cost = 0
        cpsat_changeover_cost = 0

        if self.variables.get("delay_cost_var") is not None:
            cpsat_delay_cost = cpsolvercb.Value(self.variables["delay_cost_var"])

        if self.variables.get("stock_cost_var") is not None:
            cpsat_stock_cost = cpsolvercb.Value(self.variables["stock_cost_var"])

        if "changeover_costs" in self.variables:
            for changeover_var in self.variables["changeover_costs"]:
                cpsat_changeover_cost += cpsolvercb.Value(changeover_var)
        elif "changeover_cost_var" in self.variables:
            cpsat_changeover_cost = cpsolvercb.Value(
                self.variables["changeover_cost_var"]
            )

        cpsat_total = cpsat_delay_cost + cpsat_stock_cost + cpsat_changeover_cost

        logger.info("=" * 60)
        logger.info("CP-SAT Objective Components (from objective variables):")
        logger.info(f"  Delay cost:      {cpsat_delay_cost:10.2f}")
        logger.info(f"  Stock cost:      {cpsat_stock_cost:10.2f}")
        logger.info(f"  Changeover cost: {cpsat_changeover_cost:10.2f}")
        logger.info(f"  Total:           {cpsat_total:10.2f}")
        logger.info("=" * 60)

        # Extract production decisions
        productions = []
        for item, t in self.variables["productions"]:
            qty = cpsolvercb.Value(self.variables["productions"][(item, t)])
            if qty > 0:
                productions.append(
                    ProductionDecision(item=item, period=t, quantity=qty)
                )

        # Create solution
        solution = CapacitatedMultiItemSolution(
            problem=self.problem, productions=productions
        )

        # Log actual evaluation
        objectives = self.problem.evaluate(solution)
        actual_total = solution.compute_total_cost()
        logger.info("Actual Evaluation (from problem.evaluate):")
        logger.info(f"  Inventory cost:  {objectives['inventory_cost']:10.2f}")
        logger.info(f"  Changeover cost: {objectives['changeover_cost']:10.2f}")
        logger.info(f"  Backlog cost:    {objectives['backlog_cost']:10.2f}")
        logger.info(f"  Total:           {actual_total:10.2f}")
        logger.info("=" * 60)

        # Show discrepancy if any
        discrepancy = actual_total - cpsat_total
        if abs(discrepancy) > 0.01:
            logger.warning(f"DISCREPANCY: {discrepancy:+.2f} (actual - cpsat)")
            logger.warning("  Breakdown:")
            logger.warning(
                f"    Inventory: {objectives['inventory_cost'] - cpsat_stock_cost:+.2f}"
            )
            logger.warning(
                f"    Changeover: {objectives['changeover_cost'] - cpsat_changeover_cost:+.2f}"
            )
            logger.warning(
                f"    Backlog: {objectives['backlog_cost'] - cpsat_delay_cost:+.2f}"
            )

        return solution
