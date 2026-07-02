#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Capacitated multi-item lot sizing problem with changeover costs.

This problem considers:
- Multiple item types
- Production capacity constraints
- Changeover costs when switching between items
- Inventory holding costs
- No backlog allowed (hard constraint)
- No setup costs, production costs, or setup times
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from discrete_optimization.generic_tools.do_problem import (
    EncodingRegister,
    ModeOptim,
    ObjectiveDoc,
    ObjectiveHandling,
    ObjectiveRegister,
    Solution,
    TypeObjective,
)
from discrete_optimization.generic_tools.encoding_register import ListInteger
from discrete_optimization.lotsizing import (
    GenericLotSizingProblem,
    ProductionBasedSolution,
    ProductionDecision,
)
from discrete_optimization.lotsizing.costs import (
    WithoutProductionCostsProblem,
    WithoutProductionCostsSolution,
    WithoutSetupCostsProblem,
    WithoutSetupCostsSolution,
)
from discrete_optimization.lotsizing.setup_times import (
    WithoutSetupTimesProblem,
)

logger = logging.getLogger(__name__)


class CapacitatedMultiItemSolution(
    # Override features we DON'T have (these must come before ProductionBasedSolution to override methods)
    WithoutSetupCostsSolution[int],
    WithoutProductionCostsSolution[int],
    # Production logic (automatically provides inventory/delivery computation + all GenericLotSizingSolution methods)
    ProductionBasedSolution[int],
):
    """Solution for capacitated multi-item lot sizing problem.

    Inherits from:
    - ProductionBasedSolution: Automatically computes inventory, deliveries, backlog from productions
      (which itself inherits from GenericLotSizingSolution, providing all mixin methods)
    - WithoutSetupCostsSolution: No setup costs in this problem variant
    - WithoutProductionCostsSolution: No per-unit production costs in this problem variant

    Features:
    - Backlog/delays: May be allowed or not (controlled by problem.allow_delays)
    - Changeover costs: Always present
    - Inventory costs: Always present
    - Capacity constraints: Always present

    Adds:
    - list_item_per_time: Alternative representation for binary problems
      (for each period, which item type to produce, or nb_items for idle)
    """

    problem: CapacitatedMultiItemLSP

    def __init__(
        self,
        problem: CapacitatedMultiItemLSP,
        productions: list[ProductionDecision] | None = None,
        list_item_per_time: list[int] | None = None,
    ):
        """Create a solution.

        Args:
            problem: The problem instance
            productions: List of production decisions (item, period, quantity)
            list_item_per_time: Alternative representation - for each period, which item to produce
                               (value >= nb_items or None means idle)
        """
        # If list_item_per_time is provided, convert to productions
        if productions is None and list_item_per_time is not None:
            productions = self._convert_list_to_productions(problem, list_item_per_time)
        elif productions is None:
            productions = []

        # Initialize base class with productions
        # ProductionBasedSolution will automatically compute inventory, deliveries, backlog
        super().__init__(problem=problem, productions=productions)

        # Store the list representation
        if list_item_per_time is None and productions is not None:
            # Reconstruct list_item_per_time from productions
            list_item_per_time = [problem.nb_items for _ in range(problem.horizon)]
            for prod in productions:
                list_item_per_time[prod.period] = prod.item

        self.list_item_per_time = list_item_per_time

    @staticmethod
    def _convert_list_to_productions(
        problem: CapacitatedMultiItemLSP, list_item_per_time: list[int]
    ) -> list[ProductionDecision]:
        """Convert list_item_per_time representation to productions list."""
        productions = []
        for period, item in enumerate(list_item_per_time):
            # Check if it's a valid item (not idle)
            if 0 <= item < problem.nb_items:
                # Use the available capacity as quantity
                quantity = int(problem.get_available_production_time(period))
                productions.append(
                    ProductionDecision(item=item, period=period, quantity=quantity)
                )
        return productions

    def copy(self) -> Solution:
        """Create a deep copy of the solution."""
        return CapacitatedMultiItemSolution(
            problem=self.problem,
            productions=[
                ProductionDecision(item=p.item, period=p.period, quantity=p.quantity)
                for p in self.productions
            ],
            list_item_per_time=list(self.list_item_per_time)
            if self.list_item_per_time is not None
            else None,
        )


class CapacitatedMultiItemLSP(
    # Override features we DON'T have (these must come BEFORE GenericLotSizingProblem)
    WithoutSetupCostsProblem[int],
    WithoutProductionCostsProblem[int],
    WithoutSetupTimesProblem[int],
    # Base class with all features (includes InventoryCostsProblem, ChangeoverCostsProblem, CapacityProblem, DemandsProblem, BacklogProblem)
    GenericLotSizingProblem[int],
):
    """Capacitated multi-item lot sizing problem with changeover costs.

    This is the classic lot sizing problem from CSPLib Problem 058:
    https://www.csplib.org/Problems/prob058/

    Features:
    - Multiple item types (Item = int, item indices)
    - Production capacity per period (often 1 for binary problems)
    - Changeover costs when switching between items
    - Inventory holding costs
    - Optional backlog/delays (configurable)
    - No setup costs (no cost to start production)
    - No production costs (only capacity limits, not per-unit costs)
    - No setup times (changeovers are instantaneous)

    This class composes GenericLotSizingProblem with specific feature mixins.
    Note: We DON'T inherit from SetupCostsProblem or ProductionCostsProblem
    because this problem variant doesn't have those costs.
    """

    def __init__(
        self,
        nb_items: int,
        horizon: int,
        demands: npt.NDArray[np.int_] | list[list[int]],
        capacity_machine: int,
        changeover_costs: npt.NDArray[np.int_] | list[list[int]],
        stock_cost_per_type: npt.NDArray[np.float64] | list[float],
        stock_capacity: int | None = None,
        allow_delays: bool = False,
        delay_cost_per_type: npt.NDArray[np.float64] | list[float] | None = None,
        **kwargs: Any,
    ):
        """Initialize capacitated multi-item lot sizing problem.

        Args:
            nb_items: Number of item types
            horizon: Number of time periods
            demands: Demand for each item in each period (nb_items × horizon matrix)
            capacity_machine: Production capacity per period (often 1 for binary problems)
            changeover_costs: Cost matrix for switching between items (nb_items × nb_items)
            stock_cost_per_type: Inventory holding cost per unit per period for each item type
            stock_capacity: Maximum total inventory (default: sum of all demands)
            allow_delays: Whether backlog/delays are allowed as hard constraint (default: False)
                         Note: Even if False, backlog costs are used as penalties in the objective
            delay_cost_per_type: Penalty cost per unit delay per period for each item
                                Default: [100000] * nb_items (high penalty to discourage delays)
            **kwargs: Additional parameters
        """
        # Convert to numpy arrays if needed
        if not isinstance(demands, np.ndarray):
            demands = np.array(demands, dtype=np.int64)
        if not isinstance(changeover_costs, np.ndarray):
            changeover_costs = np.array(changeover_costs, dtype=np.int64)
        if not isinstance(stock_cost_per_type, np.ndarray):
            stock_cost_per_type = np.array(stock_cost_per_type, dtype=np.float64)

        # Validate dimensions
        if demands.shape != (nb_items, horizon):
            raise ValueError(
                f"demands must have shape ({nb_items}, {horizon}), got {demands.shape}"
            )
        if changeover_costs.shape != (nb_items, nb_items):
            raise ValueError(
                f"changeover_costs must have shape ({nb_items}, {nb_items}), "
                f"got {changeover_costs.shape}"
            )

        # Default stock capacity: enough for all demands
        if stock_capacity is None:
            stock_capacity = int(np.sum(demands))

        # Store basic attributes
        self._horizon = horizon
        self._items_list = list(range(nb_items))
        # nb_items is a computed property from len(items_list)

        # Check if problem is binary (all demands are 0 or 1)
        self.is_binary = bool(np.all((demands == 0) | (demands == 1)))

        # Store for compatibility with old code
        self.items_range = range(nb_items)
        self.items_set = set(self.items_list)
        self.stock_capacity = stock_capacity

        # Store data for mixin methods
        self._demands = demands
        self._changeover_costs = changeover_costs
        self._stock_cost_per_type = stock_cost_per_type
        self._capacity_machine = capacity_machine
        self._allow_delays = allow_delays

        # Always store backlog costs (used as penalties in objective even when delays not allowed)
        if delay_cost_per_type is None:
            # Default to high penalty (same as old implementation)
            delay_cost_per_type = [100000.0] * nb_items
        if not isinstance(delay_cost_per_type, np.ndarray):
            delay_cost_per_type = np.array(delay_cost_per_type, dtype=np.float64)
        self._delay_cost_per_type = delay_cost_per_type
        self.infos = kwargs
        # Initialize "Without" mixins (no need to call GenericLotSizingProblem.__init__)
        WithoutSetupCostsProblem.__init__(self)
        WithoutProductionCostsProblem.__init__(self)
        WithoutSetupTimesProblem.__init__(self)

    @property
    def horizon(self) -> int:
        """Number of time periods."""
        return self._horizon

    @property
    def items_list(self) -> list[int]:
        """List of item indices."""
        return self._items_list

    @property
    def capacity_machine(self) -> int:
        """Production capacity per period."""
        return self._capacity_machine

    @property
    def allow_backlog(self) -> bool:
        """Whether backlog/delays are allowed."""
        return self._allow_delays

    # DemandsProblem abstract methods
    def get_demand(self, item: int, period: int) -> int:
        """Get demand for item in period."""
        return int(self._demands[item, period])

    # ChangeoverCostsProblem abstract methods
    def get_changeover_cost(self, from_item: int, to_item: int) -> float:
        """Get sequence-dependent changeover cost."""
        return float(self._changeover_costs[from_item, to_item])

    def get_changeover_array(self) -> list:
        """Get changeover costs as a 2D list."""
        return self._changeover_costs.tolist()

    # InventoryCostsProblem abstract methods
    def get_inventory_cost_per_unit(self, item: int, period: int) -> float:
        """Get inventory holding cost per unit."""
        return float(self._stock_cost_per_type[item])

    # CapacityProblem abstract methods
    def get_production_time_per_unit(self, item: int, period: int) -> float:
        """Get production time per unit (always 1 for this problem)."""
        return 1.0

    def get_available_production_time(self, period: int) -> float:
        """Get available production time in period."""
        return float(self._capacity_machine)

    # BacklogProblem abstract methods
    def is_backlog_allowed(self) -> bool:
        """Check if backlog/delays are allowed as hard constraint.

        Returns False means backlog is not allowed (constraint violation).
        However, backlog costs are still used in the objective as penalties.
        """
        return self._allow_delays

    def get_backlog_cost_per_unit(self, item: int, period: int) -> float:
        """Get backlog penalty cost per unit.

        This cost is always present in the objective (acts as penalty).
        Whether backlog is allowed as hard constraint is controlled by is_backlog_allowed().
        """
        return float(self._delay_cost_per_type[item])

    def satisfy(self, solution: CapacitatedMultiItemSolution) -> bool:
        """Check if solution satisfies all constraints.

        Uses the satisfy_partial method from GenericLotSizingProblem to check:
        - Demand satisfaction
        - Capacity constraints
        - Stock capacity
        - Unique production times (at most one item per period)

        Args:
            solution: Solution to check

        Returns:
            True if all constraints are satisfied
        """
        # Check using parent class method (checks demands, capacity, backlog)
        if not super().satisfy(solution):
            return False

        # Check stock capacity (total inventory across all items)
        total_stock = np.zeros(self.horizon, dtype=np.int64)
        for item in self.items_list:
            inventory = solution.get_inventory_level_array(item)
            total_stock += inventory

        if np.max(total_stock) > self.stock_capacity:
            logger.debug(
                f"Stock capacity exceeded: max={np.max(total_stock)}, "
                f"capacity={self.stock_capacity}"
            )
            return False

        # Check that production times are unique (at most one item per period)
        production_times = [p.period for p in solution.productions]
        if len(production_times) != len(set(production_times)):
            logger.debug("Multiple items produced in same period")
            return False

        return True

    def get_solution_type(self) -> type[Solution]:
        """Return the solution class for this problem."""
        return CapacitatedMultiItemSolution

    def get_attribute_register(self) -> EncodingRegister:
        """Return encoding register for metaheuristic solvers."""
        return EncodingRegister(
            dict_attribute_to_type={
                "list_item_per_time": ListInteger(
                    length=self.horizon, lows=0, ups=self.nb_items
                )
            }
        )

    def get_objective_register(self) -> ObjectiveRegister:
        """Return objective register."""
        return ObjectiveRegister(
            objective_sense=ModeOptim.MINIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc={
                "inventory_cost": ObjectiveDoc(
                    TypeObjective.OBJECTIVE, default_weight=1.0
                ),
                "changeover_cost": ObjectiveDoc(
                    TypeObjective.OBJECTIVE, default_weight=1.0
                ),
                "backlog_cost": ObjectiveDoc(
                    TypeObjective.PENALTY, default_weight=1000.0
                ),
                "unmet_demand": ObjectiveDoc(
                    TypeObjective.PENALTY, default_weight=100000.0
                ),
            },
        )
