#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""CP-SAT scheduling solver for capacitated multi-item lot sizing problem."""

import logging
from typing import Any

from ortools.sat.python.cp_model import CpSolverSolutionCallback

from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver
from discrete_optimization.lotsizing import ProductionDecision
from discrete_optimization.lotsizing.capacitatedmultiitem.problem import (
    CapacitatedMultiItemLSP,
    CapacitatedMultiItemSolution,
)

logger = logging.getLogger(__name__)


class CpSatSchedulingCapacitatedLotSizing(OrtoolsCpSatSolver, WarmstartMixin):
    """CP-SAT scheduling solver for capacitated multi-item lot sizing.

    Uses interval variables to model production events:
    - One interval for each demand occurrence
    - NoOverlap constraint ensures at most one production per period
    - Circuit constraint models changeover costs between consecutive productions
    - Stock cost computed as (deadline - production_time) per demand

    Based on the original scheduling solver approach.
    """

    problem: CapacitatedMultiItemLSP
    variables: dict
    deadlines: dict

    def init_model(self, **kwargs: Any) -> None:
        """Initialize the CP-SAT scheduling model."""
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        super().init_model(**kwargs)
        self._create_scheduling_vars()
        self._set_objective()

    def _create_scheduling_vars(self):
        """Create scheduling variables and constraints.

        Creates one interval for each demand occurrence (not for each time period).
        """
        self.variables = {}
        horizon = self.problem.horizon
        self.deadlines = {}
        starts = {}
        intervals = {}

        # For each item, create intervals for each demand occurrence
        for item in self.problem.items_list:
            nb = 0
            for t in range(horizon):
                demand = self.problem.get_demand(item, t)
                if demand > 0:
                    # Assuming binary demands (demand = 1)
                    self.deadlines[(item, nb)] = t

                    # Start variable: when to produce this demand
                    starts[(item, nb)] = self.cp_model.NewIntVar(
                        lb=nb,  # Can't produce before previous demands
                        ub=t if not self.problem.is_backlog_allowed() else horizon - 1,
                        name=f"start_{item}_{nb}",
                    )

                    # Interval of size 1 (one period to produce)
                    intervals[(item, nb)] = self.cp_model.NewFixedSizeIntervalVar(
                        start=starts[(item, nb)], size=1, name=f"interval_{item}_{nb}"
                    )
                    nb += 1

            # Enforce ordering within same item (earlier demands produced before later ones)
            for j in range(1, nb):
                self.cp_model.Add(starts[(item, j - 1)] < starts[(item, j)])

        # NoOverlap: at most one production per time period
        self.cp_model.AddNoOverlap(list(intervals.values()))

        self.variables["starts"] = starts
        self.variables["intervals"] = intervals

    def _create_changeover_model(self):
        """Create changeover cost variables using circuit constraint.

        Uses circuit to sequence productions, computing changeover costs only
        between consecutive productions (skipping idle periods automatically).
        """
        # Create nodes: one dummy node + one for each demand occurrence
        nodes = [("dummy", -1)]
        for key in sorted(
            self.variables["starts"].keys(), key=lambda x: self.deadlines[x]
        ):
            nodes.append(key)

        # Circuit arcs with boolean variables
        arcs = []
        next_node_vars = {}

        # Maximum gap between consecutive productions in the circuit
        # If horizon=20 and total_demands=19, there's 1 idle period total,
        # but the gap can be 2 (e.g., productions at t=17 and t=19)
        max_gap = (
            self.problem.horizon
            - sum(
                self.problem.get_total_demand(item) for item in self.problem.items_list
            )
            + 1
        )

        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i == j:
                    continue

                # Same item: enforce sequential order (skip invalid arcs)
                if nodes[i] != ("dummy", -1) and nodes[j] != ("dummy", -1):
                    item_i, nb_i = nodes[i]
                    item_j, nb_j = nodes[j]
                    if item_i == item_j and nb_j != nb_i + 1:
                        # Skip invalid orderings within same item
                        continue

                next_var = self.cp_model.NewBoolVar(name=f"next_{i}_{j}")
                next_node_vars[(i, j)] = next_var
                arcs.append((i, j, next_var))

                # Constraint: if arc (i,j) is used, enforce start_j > start_i
                if nodes[i] != ("dummy", -1) and nodes[j] != ("dummy", -1):
                    item_i, nb_i = nodes[i]
                    item_j, nb_j = nodes[j]

                    # start_j > start_i
                    self.cp_model.Add(
                        self.variables["starts"][(item_j, nb_j)]
                        > self.variables["starts"][(item_i, nb_i)]
                    ).OnlyEnforceIf(next_var)

                    # Limit gap between consecutive productions
                    self.cp_model.Add(
                        self.variables["starts"][(item_j, nb_j)]
                        <= self.variables["starts"][(item_i, nb_i)] + max_gap
                    ).OnlyEnforceIf(next_var)

        # Add circuit constraint
        self.cp_model.AddCircuit(arcs)

        # Changeover cost: sum over arcs between real productions
        cost_expr = sum(
            int(self.problem.get_changeover_cost(nodes[i][0], nodes[j][0]))
            * next_node_vars[(i, j)]
            for (i, j) in next_node_vars
            if nodes[i][0] != "dummy" and nodes[j][0] != "dummy"
        )

        self.variables["changeover_cost_expr"] = cost_expr
        self.variables["nodes"] = nodes
        self.variables["next"] = next_node_vars

    def _set_objective(self):
        """Set the objective function."""
        horizon = self.problem.horizon
        objectives = []

        # Stock cost: (deadline - start) * inventory_cost for each demand
        # This represents holding inventory from production time to delivery time
        stock_cost_expr = sum(
            int(
                self.problem.get_inventory_cost_per_unit(
                    item, self.deadlines[(item, nb)]
                )
            )
            * (self.deadlines[(item, nb)] - self.variables["starts"][(item, nb)])
            for (item, nb) in self.variables["starts"]
        )
        objectives.append(stock_cost_expr)
        self.variables["stock_cost_expr"] = stock_cost_expr

        # Delay cost: max(0, start - deadline) for backlog problems
        delay_cost_expr = None
        if self.problem.is_backlog_allowed():
            delays = {}
            for item, nb in self.variables["starts"]:
                deadline = self.deadlines[(item, nb)]
                cost_per_unit = int(
                    self.problem.get_backlog_cost_per_unit(item, deadline)
                )
                if cost_per_unit > 0:
                    delay_var = self.cp_model.NewIntVar(
                        lb=0, ub=horizon - deadline, name=f"delay_{item}_{nb}"
                    )
                    delays[(item, nb)] = delay_var
                    # delay = max(0, start - deadline)
                    self.cp_model.AddMaxEquality(
                        delay_var, [self.variables["starts"][(item, nb)] - deadline, 0]
                    )

            if delays:
                delay_cost_expr = sum(
                    int(
                        self.problem.get_backlog_cost_per_unit(
                            item, self.deadlines[(item, nb)]
                        )
                    )
                    * delays[(item, nb)]
                    for (item, nb) in delays
                )
                objectives.append(delay_cost_expr)
                self.variables["delays"] = delays

        self.variables["delay_cost_expr"] = delay_cost_expr

        # Changeover cost via circuit
        self._create_changeover_model()
        if "changeover_cost_expr" in self.variables:
            objectives.append(self.variables["changeover_cost_expr"])

        # Minimize total cost
        if objectives:
            self.cp_model.Minimize(sum(objectives))

    def set_warm_start(self, solution: CapacitatedMultiItemSolution) -> None:
        """Set warm-start hints from a solution.

        Args:
            solution: A solution to use as warm-start
        """
        self.cp_model.ClearHints()

        # Build mapping from (item, nb) to production time
        productions_by_item = {}
        for prod in solution.productions:
            if prod.item not in productions_by_item:
                productions_by_item[prod.item] = []
            productions_by_item[prod.item].append(prod.period)

        # Sort production times for each item
        for item in productions_by_item:
            productions_by_item[item].sort()

        # Map (item, nb) to production time
        for item in self.problem.items_list:
            if item in productions_by_item:
                for nb, prod_time in enumerate(productions_by_item[item]):
                    if (item, nb) in self.variables["starts"]:
                        self.cp_model.AddHint(
                            self.variables["starts"][(item, nb)], prod_time
                        )

    def retrieve_solution(
        self, cpsolvercb: CpSolverSolutionCallback
    ) -> CapacitatedMultiItemSolution:
        """Extract solution from CP-SAT solver."""
        # Extract start times for each demand
        production_times = {}
        cpsat_stock_manual = 0
        for (item, nb), start_var in self.variables["starts"].items():
            start_time = cpsolvercb.Value(start_var)
            production_times[(item, nb)] = start_time
            deadline = self.deadlines[(item, nb)]
            holding_time = deadline - start_time
            cost_per_unit = int(
                self.problem.get_inventory_cost_per_unit(item, deadline)
            )
            contribution = holding_time * cost_per_unit
            cpsat_stock_manual += contribution
            logger.debug(
                f"  Item {item}, demand {nb}: start={start_time}, deadline={deadline}, hold={holding_time}, contrib={contribution}"
            )

        # Convert to productions list (grouped by period)
        prod_by_period = {}
        for (item, nb), prod_time in production_times.items():
            if prod_time not in prod_by_period:
                prod_by_period[prod_time] = []
            prod_by_period[prod_time].append(item)

        # Create production decisions
        productions = []
        for period in sorted(prod_by_period.keys()):
            for item in prod_by_period[period]:
                # Binary problem: quantity = 1
                productions.append(
                    ProductionDecision(item=item, period=period, quantity=1)
                )

        # Create solution
        solution = CapacitatedMultiItemSolution(
            problem=self.problem, productions=productions
        )

        # Log CP-SAT objective components
        cpsat_stock_cost = (
            cpsolvercb.Value(self.variables["stock_cost_expr"])
            if "stock_cost_expr" in self.variables
            else 0
        )
        cpsat_delay_cost = (
            cpsolvercb.Value(self.variables["delay_cost_expr"])
            if self.variables.get("delay_cost_expr") is not None
            else 0
        )
        cpsat_changeover_cost = (
            cpsolvercb.Value(self.variables["changeover_cost_expr"])
            if "changeover_cost_expr" in self.variables
            else 0
        )
        cpsat_total = cpsat_stock_cost + cpsat_delay_cost + cpsat_changeover_cost

        logger.info("=" * 60)
        logger.info("CP-SAT Objective Components (from expressions):")
        logger.info(f"  Stock cost:      {cpsat_stock_cost:10.2f}")
        logger.info(f"  Stock (manual):  {cpsat_stock_manual:10.2f}")
        logger.info(f"  Delay cost:      {cpsat_delay_cost:10.2f}")
        logger.info(f"  Changeover cost: {cpsat_changeover_cost:10.2f}")
        logger.info(f"  Total:           {cpsat_total:10.2f}")
        logger.info("=" * 60)

        # Log actual evaluation
        objectives = self.problem.evaluate(solution)
        actual_total = solution.compute_total_cost()

        # Manually compute stock cost from inventory levels
        actual_stock_manual = 0
        for item in self.problem.items_list:
            for t in range(self.problem.horizon):
                inv = solution.get_inventory_level(item, t)
                if inv > 0:
                    cost = inv * self.problem.get_inventory_cost_per_unit(item, t)
                    actual_stock_manual += cost
                    logger.debug(f"  Period {t}, Item {item}: inv={inv}, cost={cost}")

        logger.info("Actual Evaluation (from problem.evaluate):")
        logger.info(f"  Inventory cost:  {objectives['inventory_cost']:10.2f}")
        logger.info(f"  Inventory (man): {actual_stock_manual:10.2f}")
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
