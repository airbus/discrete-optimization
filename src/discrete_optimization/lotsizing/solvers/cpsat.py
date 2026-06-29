#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from enum import Enum
from typing import Any

from ortools.sat.python.cp_model import (
    CpSolverSolutionCallback,
)

from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    EnumHyperparameter,
)
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver
from discrete_optimization.lotsizing.problem import (
    LotSizingProblem,
    LotSizingSolution,
    ProductionItem,
)

logger = logging.getLogger(__name__)


class ChangeoverModel(Enum):
    """Modeling approach for changeover costs in CP-SAT solver."""

    STATE_BASED = "state_based"
    TRANSITION_BASED = "transition_based"


class CpSatLotSizingSolver(OrtoolsCpSatSolver):
    problem: LotSizingProblem
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
            changeover_model: How to model changeover costs.
                - ChangeoverModel.STATE_BASED (default): Use element constraints to track last produced item.
                  More efficient, scales to large instances. O(n_items × horizon) variables.
                - ChangeoverModel.TRANSITION_BASED: Explicitly model all transitions within lookahead window.
                  Legacy approach, may not scale to large instances. O(n_items² × horizon²) variables.
            **kwargs: Additional parameters passed to parent class.
        """
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        super().init_model(**kwargs)
        self._create_main_vars()
        self._set_objective(changeover_model=kwargs["changeover_model"])

    def _create_main_vars(self):
        self.variables = {}
        total_demands_per_item = {
            item: sum(self.problem.demands[item]) for item in self.problem.items_range
        }
        horizon = self.problem.horizon
        bool_produce_type_time = {
            (item, t): self.cp_model.NewBoolVar(name=f"bool_prod_item_{item}_time_{t}")
            for item in self.problem.items_range
            for t in range(horizon)
        }
        for t in range(horizon):
            self.cp_model.add_at_most_one(
                [bool_produce_type_time[(item, t)] for item in self.problem.items_range]
            )
        if self.problem.is_binary:
            quantity_produce = bool_produce_type_time
        else:
            quantity_produce = {
                (item, t): self.cp_model.NewIntVar(
                    lb=0,
                    ub=min(self.problem.capacity_machine, total_demands_per_item[item]),
                    name=f"prod_item_{item}_time_{t}",
                )
                for item in self.problem.items_range
                for t in range(horizon)
            }
            for item, t in quantity_produce:
                self.cp_model.add(quantity_produce[(item, t)] >= 1).only_enforce_if(
                    bool_produce_type_time[(item, t)]
                )
                self.cp_model.add(quantity_produce[(item, t)] == 0).only_enforce_if(
                    bool_produce_type_time[(item, t)].Not()
                )
        delivery = {
            (item, t): self.cp_model.NewIntVar(
                lb=0,
                ub=total_demands_per_item[item],
                name=f"delivery_item_{item}_time_{t}",
            )
            for item in self.problem.items_range
            for t in range(horizon)
        }
        stocks = {
            (item, t): self.cp_model.NewIntVar(
                lb=0,
                ub=total_demands_per_item[item],
                name=f"stock_item_{item}_time_{t}",
            )
            for item in self.problem.items_range
            for t in range(horizon)
        }
        delays = {
            (item, t): self.cp_model.NewIntVar(
                lb=0,
                ub=0 if not self.problem.allow_delays else total_demands_per_item[item],
                name=f"delays_item_{item}_time_{t}",
            )
            for item in self.problem.items_range
            for t in range(horizon)
        }
        for item in self.problem.items_range:
            for t in range(horizon):
                prev_stock = 0
                prev_delay = 0
                if t > 0:
                    prev_stock = stocks[(item, t - 1)]
                    prev_delay = delays[(item, t - 1)]
                self.cp_model.add(
                    stocks[(item, t)]
                    == prev_stock + quantity_produce[(item, t)] - delivery[(item, t)]
                )
                self.cp_model.add(
                    delays[(item, t)]
                    == prev_delay + self.problem.demands[item][t] - delivery[(item, t)]
                )

        self.variables["deliveries"] = delivery
        self.variables["bool_productions"] = bool_produce_type_time
        self.variables["productions"] = quantity_produce
        self.variables["delays"] = delays
        self.variables["stocks"] = stocks

    def _create_state_based_changeover_variables(self):
        """
        Create changeover cost variables using state-based tracking.

        Tracks which item was last produced at each timestep and uses element
        constraints to look up changeover costs from the matrix.

        Much more efficient than transition-based approach:
        - O(n_items × horizon) variables instead of O(n_items² × horizon²)
        - No lookahead window complexity
        - Scales well to large instances
        """
        produce = self.variables["bool_productions"]
        horizon = self.problem.horizon
        n_items = self.problem.nb_items_type

        # State variable: which item was last produced?
        # Value = item index (0 to n_items-1), or n_items if no production yet (dummy state)
        last_item = {
            t: self.cp_model.NewIntVar(lb=0, ub=n_items, name=f"last_item_t{t}")
            for t in range(horizon)
        }

        # Initial state: dummy (no item produced yet)
        self.cp_model.add(last_item[0] == n_items)

        # Boolean: did we have any production at time t?
        has_production = {
            t: self.cp_model.NewBoolVar(name=f"has_prod_t{t}") for t in range(horizon)
        }

        for t in range(horizon):
            # has_production[t] = OR(produce[i,t] for all i)
            self.cp_model.add_max_equality(
                has_production[t],
                [produce[(item, t)] for item in self.problem.items_range],
            )

        # State evolution: last_item[t] = item produced at t, or last_item[t-1] if idle
        for t in range(1, horizon):
            # If no production at t, carry forward previous state
            self.cp_model.add(last_item[t] == last_item[t - 1]).only_enforce_if(
                has_production[t].Not()
            )

            # If producing item i at t, last_item[t] = i
            for item in self.problem.items_range:
                self.cp_model.add(last_item[t] == item).only_enforce_if(
                    produce[(item, t)]
                )

        # Changeover cost tracking
        # Flatten the changeover cost matrix with dummy row/column for "no item yet" state
        changeover_matrix_with_dummy = []
        for from_item in range(n_items + 1):  # +1 for dummy
            row = []
            for to_item in range(n_items + 1):
                if from_item == n_items or to_item == n_items:
                    # Transitions involving dummy state have 0 cost
                    row.append(0)
                else:
                    row.append(self.problem.changeover_costs[from_item][to_item])
            changeover_matrix_with_dummy.append(row)

        # Changeover cost for each production event
        max_changeover = max(max(row) for row in self.problem.changeover_costs)
        changeover_cost = {
            (item, t): self.cp_model.NewIntVar(
                lb=0, ub=max_changeover, name=f"co_cost_{item}_t{t}"
            )
            for item in self.problem.items_range
            for t in range(horizon)
        }

        for t in range(horizon):
            for item in self.problem.items_range:
                if t == 0:
                    # At t=0, last_item is dummy (n_items), so cost = 0
                    self.cp_model.add(changeover_cost[(item, t)] == 0)
                else:
                    # Use element constraint to look up cost based on previous item
                    # changeover_cost[(item, t)] = changeover_matrix_with_dummy[last_item[t-1]][item]

                    # Extract the column for current item
                    costs_for_item = [
                        changeover_matrix_with_dummy[from_item][item]
                        for from_item in range(n_items + 1)
                    ]

                    # Use AddElement: changeover_cost = costs_for_item[last_item[t-1]]
                    self.cp_model.AddElement(
                        last_item[t - 1],  # index variable
                        costs_for_item,  # array to index into
                        changeover_cost[(item, t)],  # target variable
                    )

        # Total changeover cost = sum of costs when production occurs
        total_changeover_cost = self.cp_model.NewIntVar(
            lb=0,
            ub=max_changeover * horizon,
            name="total_changeover",
        )

        # Sum up: cost is counted only when production actually happens
        cost_terms = []
        for t in range(horizon):
            for item in self.problem.items_range:
                # Create term that is changeover_cost[item,t] when produce[item,t]=1, else 0
                cost_contribution = self.cp_model.NewIntVar(
                    lb=0, ub=max_changeover, name=f"co_contrib_{item}_t{t}"
                )
                # cost_contribution = 0 if not produce, else changeover_cost
                self.cp_model.add(cost_contribution == 0).only_enforce_if(
                    produce[(item, t)].Not()
                )
                self.cp_model.add(
                    cost_contribution == changeover_cost[(item, t)]
                ).only_enforce_if(produce[(item, t)])

                cost_terms.append(cost_contribution)

        self.cp_model.add(total_changeover_cost == sum(cost_terms))

        self.variables["last_item"] = last_item
        self.variables["has_production"] = has_production
        self.variables["changeover_costs"] = changeover_cost
        self.variables["total_changeover_cost"] = total_changeover_cost

    def _create_transition_based_changeover_variables(self, lookahead: int):
        """
        Create changeover variables using explicit transition modeling (legacy approach).

        Creates binary variables for all possible transitions within a lookahead window.
        May not scale well to large instances due to O(n_items² × horizon²) complexity.

        Args:
            lookahead: Maximum time gap to consider for transitions
        """
        produce = self.variables["bool_productions"]
        max_idle_time = self.problem.horizon - sum(
            self.problem.total_demands_per_item.values()
        )
        transition = {
            ((item0, t), (item1, tprime)): self.cp_model.NewBoolVar(
                name=f"{item1, tprime}_after_{item0, t}"
            )
            for item0 in self.problem.items_range
            for item1 in self.problem.items_range
            for t in range(self.problem.horizon)
            for tprime in range(t + 1, min(t + lookahead + 1, self.problem.horizon))
        }
        has_a_production = {
            t: self.cp_model.NewBoolVar(name=f"has_a_production_{t}")
            for t in range(self.problem.horizon)
        }
        for t in range(self.problem.horizon):
            self.cp_model.add(
                sum([produce[item, t] for item in self.problem.items_range]) >= 1
            ).only_enforce_if(has_a_production[t])
            self.cp_model.add(
                sum([produce[item, t] for item in self.problem.items_range]) == 0
            ).only_enforce_if(has_a_production[t].Not())

        for orig, dest in transition:
            if dest[1] == orig[1] + 1:
                self.cp_model.add(transition[orig, dest] == 1).only_enforce_if(
                    produce[orig], produce[dest]
                )
            else:
                # Take into account the idle times
                self.cp_model.add(transition[orig, dest] == 1).only_enforce_if(
                    produce[orig],
                    produce[dest],
                    *[has_a_production[t].Not() for t in range(orig[1] + 1, dest[1])],
                )

        sum_transition = sum([transition[x] for x in transition])
        sum_production = sum([produce[x] for x in produce])
        self.cp_model.add(sum_transition == sum_production - 1)
        self.variables["transitions"] = transition

    def _set_objective(self, changeover_model: ChangeoverModel):
        horizon = self.problem.horizon
        delays = self.variables["delays"]
        stocks = self.variables["stocks"]
        objectives = []
        self.variables["objectives"] = {}
        for obj, weight in zip(
            self.params_objective_function.objectives,
            self.params_objective_function.weights,
        ):
            if obj == "delays":
                delay_obj = sum(
                    [
                        delays[(item, t)]
                        * self.problem.delay_cost_per_type_per_time_per_unit[item]
                        for t in range(horizon)
                        for item in self.problem.items_range
                    ]
                )
                self.variables["objectives"][obj] = delay_obj
                objectives.append(weight * delay_obj)
            if obj == "stock":
                stock_obj = sum(
                    [
                        stocks[(item, t)]
                        * self.problem.stock_cost_per_type_per_time_per_unit[item]
                        for t in range(horizon)
                        for item in self.problem.items_range
                    ]
                )
                self.variables["objectives"][obj] = stock_obj
                objectives.append(weight * stock_obj)
            if obj == "changeover":
                if changeover_model == ChangeoverModel.STATE_BASED:
                    self._create_state_based_changeover_variables()
                    changeover_obj = self.variables["total_changeover_cost"]
                else:  # ChangeoverModel.TRANSITION_BASED
                    self._create_transition_based_changeover_variables(
                        lookahead=min(5, self.problem.horizon)
                    )
                    transitions = self.variables["transitions"]
                    changeover_obj = sum(
                        [
                            self.problem.changeover_costs[key_0[0]][key_1[0]]
                            * transitions[key_0, key_1]
                            for key_0, key_1 in transitions
                        ]
                    )
                self.variables["objectives"][obj] = changeover_obj
                objectives.append(weight * changeover_obj)
        self.cp_model.minimize(sum(objectives))

    def retrieve_solution(
        self, cpsolvercb: CpSolverSolutionCallback
    ) -> LotSizingSolution:
        for obj in self.variables["objectives"]:
            logger.info(
                f"Objective {obj}: {cpsolvercb.value(self.variables['objectives'][obj])}"
            )
        productions = []
        deliveries = []
        for item, t in self.variables["deliveries"]:
            value = cpsolvercb.value(self.variables["deliveries"][(item, t)])
            if value > 0:
                deliveries.append(
                    ProductionItem(item_type=item, quantity=value, time=t)
                )
        for item, t in self.variables["productions"]:
            value = cpsolvercb.value(self.variables["productions"][(item, t)])
            if value > 0:
                productions.append(
                    ProductionItem(item_type=item, quantity=value, time=t)
                )
        sol = LotSizingSolution(
            problem=self.problem, productions=productions, deliveries=deliveries
        )
        if self.problem.known_bound is not None:
            logger.info(
                f"{self.aggreg_from_sol(sol) / self.problem.known_bound} relative perf"
            )
        return sol


class CpSatSchedLotSizingSolver(OrtoolsCpSatSolver, WarmstartMixin):
    """CP-SAT solver for lot sizing using scheduling model.

    This solver models lot sizing as a scheduling problem with:
    - Interval variables for each demand occurrence
    - Circuit constraints for sequencing
    - Support for warm-starting from existing solutions
    """

    problem: LotSizingProblem
    variables: dict
    deadlines: dict

    hyperparameters = [
        CategoricalHyperparameter(
            name="relax_delays",
            choices=[True, False],
            default=False,
        ),
    ]

    def init_model(self, **kwargs: Any) -> None:
        """Initialize the CP-SAT model."""
        super().init_model(**kwargs)
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        relax_delays = kwargs["relax_delays"]
        self._create_main_vars(relax_delays=relax_delays)
        self._set_objective(relax_delays=relax_delays)

    def _create_main_vars(self, relax_delays: bool = False):
        self.variables = {}
        total_demands_per_item = {
            item: sum(self.problem.demands[item]) for item in self.problem.items_range
        }
        horizon = self.problem.horizon
        self.deadlines = {}
        starts = {}
        intervals = {}
        for item in self.problem.items_range:
            demands = self.problem.demands[item]
            nb = 0
            for i in range(len(demands)):
                if demands[i] > 0:
                    # We assume it's 1 here
                    self.deadlines[(item, nb)] = i
                    starts[(item, nb)] = self.cp_model.new_int_var(
                        lb=nb,
                        ub=i if not relax_delays else horizon - 1,
                        name=f"start_{item, nb}",
                    )
                    intervals[(item, nb)] = self.cp_model.new_fixed_size_interval_var(
                        start=starts[(item, nb)], size=1, name=f"intervals_{item, nb}"
                    )
                    nb += 1
            for j in range(1, nb):
                self.cp_model.add(starts[(item, j - 1)] < starts[(item, j)])
        self.cp_model.add_no_overlap(list(intervals.values()))
        self.variables["starts"] = starts
        self.variables["intervals"] = intervals

    def _set_objective(self, relax_delays: bool = False):
        horizon = self.problem.horizon
        objectives = []
        self.variables["objectives"] = {}
        for obj, weight in zip(
            self.params_objective_function.objectives,
            self.params_objective_function.weights,
        ):
            if obj == "delays":
                # delay_obj = sum([self.variables["starts"][k]
                #                 * self.problem.delay_cost_per_type_per_time_per_unit[item]
                #                 for t in range(horizon)
                #                for item in self.problem.items_range])
                if relax_delays:
                    delays = {
                        (item, nb): self.cp_model.new_int_var(
                            lb=0,
                            ub=self.problem.horizon - self.deadlines[(item, nb)],
                            name=f"delay_{item, nb}",
                        )
                        for item, nb in self.variables["starts"]
                    }
                    self.variables["delays"] = delays
                    for item, nb in delays:
                        self.cp_model.add_max_equality(
                            delays[(item, nb)],
                            [
                                self.variables["starts"][(item, nb)]
                                - self.deadlines[(item, nb)],
                                0,
                            ],
                        )
                    self.variables["objectives"][obj] = sum(
                        [
                            self.problem.delay_cost_per_type_per_time_per_unit[item]
                            * delays[(item, nb)]
                            for item, nb in delays
                        ]
                    )
                else:
                    self.variables["objectives"][obj] = 0
                print("weights on delay", weight)
                objectives.append(weight * self.variables["objectives"][obj])
            if obj == "stock":
                stock_obj = sum(
                    [
                        (self.deadlines[k] - self.variables["starts"][k])
                        * self.problem.stock_cost_per_type_per_time_per_unit[k[0]]
                        for k in self.deadlines
                    ]
                )
                self.variables["objectives"][obj] = stock_obj
                objectives.append(weight * stock_obj)
            if obj == "changeover":
                self._create_changeover_model()
                objectives.append(weight * self.variables["objectives"]["changeover"])
        self.cp_model.minimize(sum(objectives))

    def _create_changeover_model(self):
        keys = sorted(self.variables["starts"].keys(), key=lambda x: self.deadlines[x])
        nodes = [("dummy", 0)] + keys  #
        arcs = []
        next_node_vars = {}
        max_idle_time = self.problem.horizon - sum(
            self.problem.total_demands_per_item.values()
        )
        for i in range(len(nodes)):
            item_i, ind_i = nodes[i]
            for j in range(len(nodes)):
                if i == j:
                    continue
                item_j, ind_j = nodes[j]
                if item_i == item_j and ind_j != ind_i + 1:
                    # convention order..
                    continue
                next_node_vars[(i, j)] = self.cp_model.new_bool_var(name=f"next_{i, j}")
                arcs.append((i, j, next_node_vars[(i, j)]))
                if item_i == "dummy" or item_j == "dummy":
                    continue
                self.cp_model.add(
                    self.variables["starts"][(item_j, ind_j)]
                    > self.variables["starts"][(item_i, ind_i)]
                ).only_enforce_if(next_node_vars[(i, j)])
                self.cp_model.add(
                    self.variables["starts"][(item_j, ind_j)]
                    <= self.variables["starts"][(item_i, ind_i)] + max_idle_time
                ).only_enforce_if(next_node_vars[(i, j)])
        self.cp_model.add_circuit(arcs)
        cost = sum(
            [
                self.problem.changeover_costs[nodes[i][0]][nodes[j][0]]
                * next_node_vars[(i, j)]
                for i, j in next_node_vars
                if nodes[i][0] != "dummy" and nodes[j][0] != "dummy"
            ]
        )
        self.variables["objectives"]["changeover"] = cost
        self.variables["nodes_convention"] = nodes
        self.variables["next"] = next_node_vars

    def set_warm_start(self, solution: LotSizingSolution) -> None:
        """Set warm-start hints from a LotSizingSolution.

        Args:
            solution: A LotSizingSolution to use as warm-start
        """
        # Build mapping from (item_type, occurrence) to production time
        # Group productions by item type and sort by time
        productions_by_type = {}
        for prod in solution.productions:
            if prod.item_type not in productions_by_type:
                productions_by_type[prod.item_type] = []
            productions_by_type[prod.item_type].append(prod.time)

        # Sort production times for each type
        for item_type in productions_by_type:
            productions_by_type[item_type].sort()

        # Map (item_type, nb) to production time
        production_times = {}
        for item_type in productions_by_type:
            for nb in range(len(productions_by_type[item_type])):
                production_times[(item_type, nb)] = productions_by_type[item_type][nb]

        # Set hints for start variables
        self.cp_model.ClearHints()
        num_hints = 0
        for key, start_var in self.variables["starts"].items():
            if key in production_times:
                hint_value = production_times[key]
                # Ensure hint is within variable bounds
                self.cp_model.AddHint(start_var, hint_value)
                num_hints += 1
                logger.debug(f"Warm-start hint: {key} -> {hint_value}")
        if "next" in self.variables:
            nodes_convention = self.variables["nodes_convention"]
            node_to_index = {
                nodes_convention[i]: i for i in range(len(nodes_convention))
            }
            sorted_production_full = sorted(solution.productions, key=lambda p: p.time)
            sorted_node = [0]
            count_per_item_type = {item_type: 0 for item_type in productions_by_type}
            for p in sorted_production_full:
                sorted_node.append(
                    node_to_index[(p.item_type, count_per_item_type[p.item_type])]
                )
                count_per_item_type[p.item_type] += 1
            arcs_actives = set(
                [(n0, n1) for n0, n1 in zip(sorted_node[:-1], sorted_node[1:])]
                + [(sorted_node[-1], 0)]
            )
            for i, j in self.variables["next"]:
                if (i, j) in arcs_actives:
                    self.cp_model.AddHint(self.variables["next"][(i, j)], 1)
                else:
                    self.cp_model.AddHint(self.variables["next"][(i, j)], 0)
        if "delays" in self.variables:
            for key in self.variables["delays"]:
                delay = production_times[key]
                deadline = self.deadlines[key]
                if delay > deadline:
                    self.cp_model.AddHint(
                        self.variables["delays"][key], delay - deadline
                    )
                else:
                    self.cp_model.AddHint(self.variables["delays"][key], 0)

    def retrieve_solution(
        self, cpsolvercb: CpSolverSolutionCallback
    ) -> LotSizingSolution:
        for obj in self.variables["objectives"]:
            logger.info(
                f"Objective {obj}: {cpsolvercb.value(self.variables['objectives'][obj])}"
            )
        productions = []
        deliveries = []
        for item, index_item in self.variables["starts"]:
            value = cpsolvercb.value(self.variables["starts"][(item, index_item)])
            productions.append(ProductionItem(item_type=item, quantity=1, time=value))
            deliveries.append(
                ProductionItem(
                    item_type=item, quantity=1, time=self.deadlines[item, index_item]
                )
            )
        sol = LotSizingSolution(
            problem=self.problem, productions=productions, deliveries=deliveries
        )
        if self.problem.known_bound is not None:
            logger.info(
                f"{self.aggreg_from_sol(sol) / self.problem.known_bound} relative perf"
            )
        return sol
