#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Generic lot sizing problem composing all mixins.

This module provides GenericLotSizingProblem and GenericLotSizingSolution that
combine all feature mixins, similar to GenericSchedulingProblem in generic_tasks_tools.

This generic class encompasses all lot sizing variants by composing mixins.
Specific variants can disable features using "Without" mixins.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Generic, TypeVar

# Generic type for items/products
# Usually int, but could be string, enum, or any hashable type
Item = TypeVar("Item", bound=Hashable)
from typing import Generic

from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveDoc,
    ObjectiveHandling,
    ObjectiveRegister,
    TypeObjective,
)
from discrete_optimization.lotsizing.backlog import (
    BacklogProblem,
    BacklogSolution,
)
from discrete_optimization.lotsizing.base import Item
from discrete_optimization.lotsizing.changeover import (
    ChangeoverCostsProblem,
    ChangeoverCostsSolution,
)
from discrete_optimization.lotsizing.costs import (
    InventoryCostsProblem,
    InventoryCostsSolution,
    ProductionCostsProblem,
    ProductionCostsSolution,
    SetupCostsProblem,
    SetupCostsSolution,
)
from discrete_optimization.lotsizing.setup_times import (
    SetupTimesProblem,
    SetupTimesSolution,
)


class GenericLotSizingProblem(
    # Costs
    SetupCostsProblem[Item],
    ProductionCostsProblem[Item],
    InventoryCostsProblem[Item],
    # Optional features
    BacklogProblem[Item],
    ChangeoverCostsProblem[Item],
    # Capacity (extends DemandsProblem which extends LotSizingProblem)
    # SetupTimesProblem already extends CapacityProblem, so we get it transitively
    SetupTimesProblem[Item],
    Generic[Item],
):
    """Generic lot sizing problem with ALL optional features.

    Similar to GenericSchedulingProblem in generic_tasks_tools, this class
    encompasses all lot sizing variants by composing mixins:

    - **Single-item or multi-item**: Controlled by items_list
    - **Uncapacitated or capacitated**: Use WithoutCapacityProblem for uncapacitated
    - **With or without backlog**: Use WithoutBacklogProblem if backlog not allowed
    - **With or without setup times**: Use WithoutSetupTimesProblem if setup times don't consume capacity
    - **With or without changeover costs**: Use WithoutChangeoverCostsProblem for sequence-independent problems

    Each feature can be disabled using the corresponding "Without" mixin.

    Example variants:
    - ULSP (Uncapacitated Lot-Sizing Problem): Use WithoutCapacityProblem
    - CLSP (Capacitated Lot-Sizing Problem): Use CapacityProblem
    - CLSP with setup times: Use SetupTimesProblem
    - CLSP with backlog: Use BacklogProblem with is_backlog_allowed() = True
    """

    def get_objective_register(self) -> ObjectiveRegister:
        """Define objectives for lot sizing problems.

        Returns:
            Objective register with setup, production, inventory, backlog, and changeover costs
        """
        return ObjectiveRegister(
            objective_sense=ModeOptim.MINIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc={
                "setup_cost": ObjectiveDoc(
                    type=TypeObjective.OBJECTIVE, default_weight=1.0
                ),
                "production_cost": ObjectiveDoc(
                    type=TypeObjective.OBJECTIVE, default_weight=1.0
                ),
                "inventory_cost": ObjectiveDoc(
                    type=TypeObjective.OBJECTIVE, default_weight=1.0
                ),
                "backlog_cost": ObjectiveDoc(
                    type=TypeObjective.OBJECTIVE, default_weight=1.0
                ),
                "changeover_cost": ObjectiveDoc(
                    type=TypeObjective.OBJECTIVE, default_weight=1.0
                ),
                # Penalty for unmet demand (useful when backlog not allowed)
                "unmet_demand": ObjectiveDoc(
                    type=TypeObjective.PENALTY, default_weight=1000.0
                ),
            },
        )

    def evaluate(self, variable: "GenericLotSizingSolution") -> dict[str, float]:
        """Evaluate solution and compute all objective components.

        Args:
            variable: Solution to evaluate

        Returns:
            Dictionary with objective values
        """
        return {
            "setup_cost": variable.compute_total_setup_cost(),
            "production_cost": variable.compute_total_production_cost(),
            "inventory_cost": variable.compute_total_inventory_cost(),
            "backlog_cost": variable.compute_total_backlog_cost(),
            "changeover_cost": variable.compute_total_changeover_cost(),
            "unmet_demand": float(variable.get_total_unmet_demand()),
        }

    def satisfy(self, variable: "GenericLotSizingSolution") -> bool:
        """Check all constraints.

        Args:
            variable: Solution to check

        Returns:
            True if all constraints satisfied, False otherwise
        """
        return self.satisfy_partial(variable)

    def satisfy_partial(
        self,
        variable: "GenericLotSizingSolution",
        demands: bool = True,
        capacity: bool = True,
        backlog: bool = True,
    ) -> bool:
        """Partial constraint checking.

        One can switch off some checks by setting the corresponding parameter to False.
        Useful for debugging or progressive solution construction.

        Args:
            variable: Solution to check
            demands: Check demand satisfaction
            capacity: Check capacity constraints
            backlog: Check backlog constraints

        Returns:
            True if selected constraints satisfied, False otherwise
        """
        return (
            # Demand satisfaction
            (
                not demands
                or variable.check_demand_satisfaction(
                    allow_delays=self.is_backlog_allowed()
                )
            )
            # Capacity constraints
            and (not capacity or variable.check_capacity_constraints())
            # Backlog constraints
            and (not backlog or variable.check_backlog_constraints())
        )


class GenericLotSizingSolution(
    # All solution mixins
    SetupCostsSolution[Item],
    ProductionCostsSolution[Item],
    InventoryCostsSolution[Item],
    BacklogSolution[Item],
    SetupTimesSolution[Item],
    ChangeoverCostsSolution[Item],
    # SetupTimesSolution extends CapacitySolution
    # CapacitySolution extends DemandsSolution
    # DemandsSolution extends LotSizingSolution
    Generic[Item],
):
    """Generic lot sizing solution corresponding to GenericLotSizingProblem.

    This solution class combines all mixin solution classes, providing:
    - Production and setup tracking
    - Inventory and delivery computation
    - Backlog tracking
    - Cost computation for all components
    - Constraint checking

    Concrete implementations should inherit from this and provide:
    - get_production_quantity()
    - has_setup()
    - get_delivery_quantity()
    - get_inventory_level()
    - get_backlog_quantity()
    - get_production_sequence()
    """

    problem: GenericLotSizingProblem[Item]

    def compute_total_cost(self) -> float:
        """Compute total cost of all components.

        Returns:
            Sum of all cost components
        """
        return (
            self.compute_total_setup_cost()
            + self.compute_total_production_cost()
            + self.compute_total_inventory_cost()
            + self.compute_total_backlog_cost()
            + self.compute_total_changeover_cost()
        )

    def get_cost_evolution(self) -> dict[str, list[float]]:
        """Get cumulative cost evolution over time for all components.

        Returns a dictionary with cumulative costs for each period:
        - 'inventory': Cumulative inventory holding costs
        - 'backlog': Cumulative backlog/delay costs
        - 'setup': Cumulative setup costs
        - 'production': Cumulative production costs
        - 'changeover': Cumulative changeover costs
        - 'total': Cumulative total cost

        Returns:
            Dictionary mapping cost component names to lists of cumulative costs
        """
        horizon = self.problem.horizon
        items = self.problem.items_list

        # Initialize cumulative totals
        costs = {
            "inventory": [],
            "backlog": [],
            "setup": [],
            "production": [],
            "changeover": [],
            "total": [],
        }

        total_inv = 0.0
        total_backlog = 0.0
        total_setup = 0.0
        total_production = 0.0
        total_changeover = 0.0

        # Track previous item for changeover detection
        prev_item_produced = None

        for t in range(horizon):
            # Accumulate costs for this period
            for item in items:
                # Inventory cost
                inv_level = self.get_inventory_level(item, t)
                total_inv += inv_level * self.problem.get_inventory_cost_per_unit(
                    item, t
                )

                # Backlog cost
                backlog = self.get_backlog_quantity(item, t)
                total_backlog += backlog * self.problem.get_backlog_cost_per_unit(
                    item, t
                )

                # Setup cost
                if self.has_setup(item, t):
                    total_setup += self.problem.get_setup_cost(item, t)

                # Production cost
                qty = self.get_production_quantity(item, t)
                total_production += qty * self.problem.get_production_cost_per_unit(
                    item, t
                )

            # Changeover cost - detect item transitions
            curr_item_produced = None
            for item in items:
                if self.get_production_quantity(item, t) > 0:
                    curr_item_produced = item
                    break

            if prev_item_produced is not None and curr_item_produced is not None:
                if prev_item_produced != curr_item_produced:
                    total_changeover += self.problem.get_changeover_cost(
                        prev_item_produced, curr_item_produced
                    )

            prev_item_produced = curr_item_produced

            # Store cumulative costs for this period
            costs["inventory"].append(total_inv)
            costs["backlog"].append(total_backlog)
            costs["setup"].append(total_setup)
            costs["production"].append(total_production)
            costs["changeover"].append(total_changeover)
            costs["total"].append(
                total_inv
                + total_backlog
                + total_setup
                + total_production
                + total_changeover
            )

        return costs
