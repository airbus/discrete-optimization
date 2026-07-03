#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Capacitated lot sizing problem with setup times.

This problem adds setup times to the CLSP formulation (page 11, slides_313.pdf):
- Setup time τ_it is consumed when production setup occurs
- Capacity constraint: Σ_i (p_it·X_it + τ_it·Y_it) ≤ h_t
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
    WithoutChangeoverCostsProblem,
    WithoutChangeoverCostsSolution,
    WithoutProductionCostsProblem,
    WithoutProductionCostsSolution,
    WithoutSetupCostsProblem,
    WithoutSetupCostsSolution,
    WithoutStockLimitsProblem,
    WithoutStockLimitsSolution,
    WithParallelProductionProblem,
    WithParallelProductionSolution,
)
from discrete_optimization.lotsizing.base import Item

logger = logging.getLogger(__name__)


class CapacitatedSetupTimesSolution(
    # Override features we DON'T have (must come first)
    WithoutSetupCostsSolution[int],
    WithoutProductionCostsSolution[int],
    WithoutChangeoverCostsSolution[int],
    WithoutStockLimitsSolution[int],
    WithParallelProductionSolution[int],
    # Production logic (auto-computes inventory/delivery/backlog)
    # ProductionBasedSolution inherits from GenericLotSizingSolution,
    # which already includes SetupTimesSolution - so we get setup time handling automatically!
    ProductionBasedSolution[int],
):
    """Solution for capacitated lot sizing with setup times.

    Inherits from:
    - ProductionBasedSolution: Auto-computes inventory, deliveries, backlog
      - Via GenericLotSizingSolution: Already includes SetupTimesSolution!
    - WithoutSetupCostsSolution: No setup costs
    - WithoutProductionCostsSolution: No per-unit production costs

    Features:
    - Setup times consuming capacity (automatic via SetupTimesSolution mixin!)
    - Changeover costs
    - Inventory costs
    - Backlog/delays (configurable)

    Note: SetupTimesSolution.get_total_production_time_used() is inherited automatically
    through ProductionBasedSolution -> GenericLotSizingSolution -> SetupTimesSolution.
    No need to override - the mixin does it for us!
    """

    problem: CapacitatedSetupTimesLSP


class CapacitatedSetupTimesLSP(
    # Override features we DON'T have (must come first)
    WithoutSetupCostsProblem[int],
    WithoutProductionCostsProblem[int],
    WithoutChangeoverCostsProblem[int],
    WithoutStockLimitsProblem[int],
    WithParallelProductionProblem[int],
    # Base class with all features (SetupTimesProblem is included via CapacityProblem in GenericLotSizingProblem)
    GenericLotSizingProblem[int],
):
    """Capacitated lot sizing problem with setup times.

    Based on CLSP with setup times formulation (page 11, slides_313.pdf):
    - New parameter: τ_it = Fixed setup time for item i in period t
    - Modified capacity constraint: Σ_i (p_it·X_it + τ_it·Y_it) ≤ h_t

    Features:
    - Multiple item types
    - Production capacity per period
    - Setup times consuming capacity
    - Changeover costs when switching items
    - Inventory holding costs
    - Optional backlog/delays
    - No setup costs or production costs
    """

    def get_stock_limit(self, item: Item, period: int) -> int | float:
        if self.stock_capacity is None:
            return float("inf")
        return self.stock_capacity

    def allows_parallel_production(self) -> bool:
        return True

    def __init__(
        self,
        nb_items: int,
        horizon: int,
        demands: npt.NDArray[np.int_] | list[list[int]],
        capacity_machine: int | float,
        setup_times: npt.NDArray[np.float64] | list[list[float]],
        stock_cost_per_type: npt.NDArray[np.float64] | list[float],
        stock_capacity: int | None = None,
        allow_delays: bool = False,
        delay_cost_per_type: npt.NDArray[np.float64] | list[float] | None = None,
        **kwargs: Any,
    ):
        """Initialize capacitated lot sizing problem with setup times.

        Args:
            nb_items: Number of item types
            horizon: Number of time periods
            demands: Demand for each item in each period (nb_items × horizon matrix)
            capacity_machine: Production capacity per period
            setup_times: Setup time τ_it for each item in each period (nb_items × horizon matrix)
            changeover_costs: Cost matrix for switching between items (nb_items × nb_items)
            stock_cost_per_type: Inventory holding cost per unit per period for each item type
            stock_capacity: Maximum total inventory (default: sum of all demands)
            allow_delays: Whether backlog/delays are allowed (default: False)
            delay_cost_per_type: Penalty cost per unit delay per period for each item
                                Default: [100000] * nb_items
            **kwargs: Additional parameters
        """
        # Convert to numpy arrays if needed
        if not isinstance(demands, np.ndarray):
            demands = np.array(demands, dtype=np.int64)
        if not isinstance(setup_times, np.ndarray):
            setup_times = np.array(setup_times, dtype=np.float64)
        if not isinstance(stock_cost_per_type, np.ndarray):
            stock_cost_per_type = np.array(stock_cost_per_type, dtype=np.float64)

        # Validate dimensions
        if demands.shape != (nb_items, horizon):
            raise ValueError(
                f"demands must have shape ({nb_items}, {horizon}), got {demands.shape}"
            )
        if setup_times.shape != (nb_items, horizon):
            raise ValueError(
                f"setup_times must have shape ({nb_items}, {horizon}), got {setup_times.shape}"
            )
        # Default stock capacity: enough for all demands
        if stock_capacity is None:
            stock_capacity = int(np.sum(demands))

        # Store basic attributes
        self._horizon = horizon
        self._items_list = list(range(nb_items))

        # Check if problem is binary (all demands are 0 or 1)
        self.is_binary = bool(np.all((demands == 0) | (demands == 1)))

        # Store for compatibility
        self.items_range = range(nb_items)
        self.items_set = set(self.items_list)
        self.stock_capacity = stock_capacity

        # Store data for mixin methods
        self._demands = demands
        self._setup_times = setup_times
        self._stock_cost_per_type = stock_cost_per_type
        self._capacity_machine = float(capacity_machine)
        self._allow_delays = allow_delays

        # Always store backlog costs (used as penalties even when delays not allowed)
        if delay_cost_per_type is None:
            delay_cost_per_type = [100000.0] * nb_items
        if not isinstance(delay_cost_per_type, np.ndarray):
            delay_cost_per_type = np.array(delay_cost_per_type, dtype=np.float64)
        self._delay_cost_per_type = delay_cost_per_type
        self.infos = kwargs

        # Initialize "Without" mixins
        WithoutSetupCostsProblem.__init__(self)
        WithoutProductionCostsProblem.__init__(self)

    @property
    def horizon(self) -> int:
        """Number of time periods."""
        return self._horizon

    @property
    def items_list(self) -> list[int]:
        """List of item indices."""
        return self._items_list

    @property
    def capacity_machine(self) -> float:
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

    # SetupTimesProblem abstract methods
    def get_setup_time(self, item: int, period: int) -> float:
        """Get setup time τ_it for item in period."""
        return float(self._setup_times[item, period])

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
        return self._capacity_machine

    # BacklogProblem abstract methods
    def is_backlog_allowed(self) -> bool:
        """Check if backlog/delays are allowed as hard constraint."""
        return self._allow_delays

    def get_backlog_cost_per_unit(self, item: int, period: int) -> float:
        """Get backlog penalty cost per unit."""
        return float(self._delay_cost_per_type[item])

    def satisfy(self, solution: CapacitatedSetupTimesSolution) -> bool:
        """Check if solution satisfies all constraints.

        Uses satisfy_partial from GenericLotSizingProblem to check:
        - Demand satisfaction
        - Capacity constraints (including setup times)
        - Stock capacity
        - Unique production times

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
        return True

    def get_solution_type(self) -> type[Solution]:
        """Return the solution class for this problem."""
        return CapacitatedSetupTimesSolution

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
