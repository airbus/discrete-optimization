#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""OptalCP solver for lot sizing problem using scheduling model."""

from __future__ import annotations

import logging
from typing import Any, Optional

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.hub_solver.optal.optalcp_tools import (
    OptalCpSolver,
)
from discrete_optimization.lotsizing.problem import (
    LotSizingProblem,
    LotSizingSolution,
    ProductionItem,
)

try:
    import optalcp as cp
except ImportError:
    cp = None
    optalcp_available = False
else:
    optalcp_available = True

logger = logging.getLogger(__name__)


class OptalSchedLotSizingSolver(OptalCpSolver):
    """OptalCP solver for lot sizing using scheduling-based model.

    This solver models the lot sizing problem as a scheduling problem where:
    - Each demand occurrence (item_type, occurrence_nb) is a task to schedule
    - Tasks have deadlines (time when demand occurs)
    - Precedence constraints enforce ordering of same-type items
    - Changeover costs depend on sequence of item types
    - Stock cost = producing before deadline
    - Delay cost = producing after deadline

    Model is similar to DpSchedLotSizingSolver but uses OptalCP instead of DIDPPy.
    """

    problem: LotSizingProblem

    def __init__(
        self,
        problem: LotSizingProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}
        self.deadlines = {}
        self.all_items = []

    def init_model(self, **kwargs: Any) -> None:
        """Initialize the OptalCP model for lot sizing.

        Creates:
        - Interval variables for each production item (item_type, occurrence_nb)
        - No-overlap constraint for machine capacity
        - Precedence constraints for same-type items
        - Objective: minimize changeover + delay + stock costs
        """
        self.cp_model = cp.Model()

        # Build list of items to schedule: (item_type, occurrence_nb)
        # For each item type, create one item per demand occurrence
        self.deadlines = {}
        for item in self.problem.items_range:
            demands = self.problem.demands[item]
            nb = 0
            for t in range(len(demands)):
                if demands[t] > 0:
                    # Assume binary demands (quantity=1)
                    self.deadlines[(item, nb)] = t
                    nb += 1

        self.all_items = sorted(self.deadlines.keys(), key=lambda x: self.deadlines[x])
        logger.info(f"Created {len(self.all_items)} items to schedule")

        # Create interval variables for each item (time dimension)
        # Each item takes 1 time unit to produce
        intervals_time = {}
        for item, nb in self.all_items:
            deadline = self.deadlines[(item, nb)]
            # Start can be from 0 to horizon
            # Length is always 1 (unit production time)
            intervals_time[(item, nb)] = self.cp_model.interval_var(
                start=(0, self.problem.horizon),
                end=(0, self.problem.horizon),  # self.problem.horizon),
                length=1,
                optional=False,
                name=f"prod_{item}_{nb}",
            )

        # No-overlap constraint: only one item can be produced at a time
        # self.cp_model.no_overlap([intervals_time[item] for item in self.all_items])

        # Precedence constraints: (item, k-1) must finish before (item, k) starts
        for item_type in self.problem.items_range:
            # Get all items of this type
            items_of_type = [(i, nb) for i, nb in self.all_items if i == item_type]
            items_of_type.sort(key=lambda x: x[1])  # Sort by occurrence number

            for idx in range(len(items_of_type) - 1):
                current_item = items_of_type[idx]
                next_item = items_of_type[idx + 1]
                # Current must end before next starts
                self.cp_model.end_before_start(
                    intervals_time[current_item], intervals_time[next_item]
                )

        # Create time sequence
        interval_list_time = [intervals_time[item] for item in self.all_items]
        item_types = [item[0] for item in self.all_items]  # Extract item type for each

        seq_time = self.cp_model.sequence_var(interval_list_time, item_types)
        seq_time.no_overlap()

        # Create cost-space sequence for changeover costs
        # This follows the dual-sequence approach from ovensched solver:
        # - Create parallel intervals in "cost space" (zero-length)
        # - Apply changeover_costs as transitions in cost space
        # - Use _same_sequence to link time and cost sequences
        # - Total changeover cost = makespan of cost sequence

        # Estimate maximum total changeover cost
        max_changeover_cost = sum(
            max(row) for row in self.problem.changeover_costs
        ) * len(self.all_items)

        intervals_cost = {}
        for item, nb in self.all_items:
            # Position in cost-space = cumulative changeover cost
            intervals_cost[(item, nb)] = self.cp_model.interval_var(
                start=(0, max_changeover_cost),
                end=(0, max_changeover_cost),
                length=1,  # Point in cost-space, not time
                optional=False,
                name=f"cost_{item}_{nb}",
            )

        # Create cost sequence with same ordering as time sequence
        interval_list_cost = [intervals_cost[item] for item in self.all_items]
        seq_cost = self.cp_model.sequence_var(interval_list_cost, item_types)
        for item in self.all_items:
            for item2 in self.all_items:
                self.cp_model.enforce(
                    self.cp_model.implies(
                        self.cp_model.start(intervals_time[item])
                        < self.cp_model.start(intervals_time[item2]),
                        self.cp_model.start(intervals_cost[item])
                        < self.cp_model.start(intervals_cost[item2]),
                    )
                )
        # Apply changeover costs as transitions in cost space
        seq_cost.no_overlap(self.problem.changeover_costs)
        # for item in self.all_items:
        #    self.cp_model.enforce(self.cp_model.position(intervals_cost[item], seq_cost)==
        #                          self.cp_model.position(intervals_time[item], seq_time))
        # Link time and cost sequences: they must have identical ordering
        # self.cp_model.enforce(self.cp_model._same_sequence(seq_time, seq_cost))

        # Build objective function
        objective_terms = []

        # 1. Delay costs: cost for producing after deadline
        for item, nb in self.all_items:
            deadline = self.deadlines[(item, nb)]
            delay_cost = self.problem.delay_cost_per_type_per_time_per_unit[item]

            # delay = max(0, end_time - deadline)
            delay = self.cp_model.max2(
                self.cp_model.start(intervals_time[(item, nb)]) - deadline, 0
            )
            objective_terms.append(delay * delay_cost)

        # 2. Stock costs: cost for producing before deadline
        stocks = []
        for item, nb in self.all_items:
            deadline = self.deadlines[(item, nb)]
            stock_cost = self.problem.stock_cost_per_type_per_time_per_unit[item]

            # stock = max(0, deadline - end_time)
            # Stock holding time when we produce early
            stock = self.cp_model.max2(
                deadline - self.cp_model.start(intervals_time[(item, nb)]), 0
            )
            stocks.append(stock * stock_cost)
            objective_terms.append(stock * stock_cost)
        # self.cp_model.enforce(self.cp_model.sum(stocks)<=9000)
        # 3. Changeover costs: extracted from cost-space sequence
        # Total changeover cost = end position of last item in cost-space
        # Since intervals have length=0, the end position = cumulative changeover cost
        all_cost_ends = [
            self.cp_model.end(intervals_cost[item]) for item in self.all_items
        ]
        total_changeover_cost = self.cp_model.max(all_cost_ends)
        objective_terms.append(total_changeover_cost)

        # Minimize total cost (delay + stock + changeover)
        total_cost = self.cp_model.sum(objective_terms) if objective_terms else 0
        self.cp_model.minimize(total_cost)

        # Store variables
        self.variables["intervals_time"] = intervals_time
        self.variables["intervals_cost"] = intervals_cost
        self.variables["seq_time"] = seq_time
        self.variables["seq_cost"] = seq_cost

    def retrieve_solution(self, result: "cp.SolveResult") -> Solution:
        """Extract solution from OptalCP result.

        Args:
            result: OptalCP solve result

        Returns:
            LotSizingSolution with production schedule
        """
        if result.solution is None:
            # Return empty solution if no solution found
            return LotSizingSolution(
                problem=self.problem, productions=[], deliveries=[]
            )

        intervals_time = self.variables["intervals_time"]
        intervals_cost = self.variables["intervals_cost"]
        productions = []
        deliveries = []

        # Extract production times and sort by start time to get the sequence
        production_schedule = []
        for item, nb in self.all_items:
            time_interval_value = result.solution.get_value(intervals_time[(item, nb)])
            cost_interval_value = result.solution.get_value(intervals_cost[(item, nb)])
            production_schedule.append(
                {
                    "item": (item, nb),
                    "item_type": item,
                    "production_time": time_interval_value[0],
                    "cost_start": cost_interval_value[0],
                    "cost_end": cost_interval_value[1],
                }
            )

        # Sort by production time to get the actual sequence
        production_schedule.sort(key=lambda x: x["production_time"])

        # # Log cost intervals and verify changeover constraints
        # logger.info("=== Cost-space intervals (in production order) ===")
        # for i, entry in enumerate(production_schedule):
        #     logger.info(f"  [{i}] Item {entry['item']}: type={entry['item_type']}, "
        #                f"prod_time={entry['production_time']}, "
        #                f"cost=[{entry['cost_start']}, {entry['cost_end']}]")
        #
        # # Verify changeover costs between consecutive items
        # logger.info("\n=== Verifying changeover costs ===")
        # for i in range(len(production_schedule) - 1):
        #     curr = production_schedule[i]
        #     next_item = production_schedule[i + 1]
        #
        #     # Gap in cost space
        #     gap = next_item['cost_start'] - curr['cost_end']
        #
        #     # Expected changeover cost from matrix
        #     expected_changeover = self.problem.changeover_costs[curr['item_type']][next_item['item_type']]
        #
        #     match = "✓" if abs(gap - expected_changeover) < 0.001 else "✗"
        #     logger.info(f"  {match} Item {curr['item']} (type {curr['item_type']}) -> "
        #                f"Item {next_item['item']} (type {next_item['item_type']}): "
        #                f"gap={gap}, expected={expected_changeover}")

        all_cost_ends = [entry["cost_end"] for entry in production_schedule]
        total_changeover = max(all_cost_ends) if all_cost_ends else 0
        logger.info(f"\n  Total changeover cost (max of ends): {total_changeover}")
        logger.info("============================\n")

        # Extract production schedule
        for entry in production_schedule:
            item, nb = entry["item"]
            production_time = entry["production_time"]
            deadline = self.deadlines[(item, nb)]

            productions.append(
                ProductionItem(
                    item_type=item,
                    quantity=1,  # Binary demands
                    time=production_time,
                )
            )

            # Delivery happens at max(production_time + 1, deadline)
            # Production finishes at production_time + 1
            delivery_time = max(production_time, deadline)
            deliveries.append(
                ProductionItem(item_type=item, quantity=1, time=delivery_time)
            )

        return LotSizingSolution(
            problem=self.problem, productions=productions, deliveries=deliveries
        )
