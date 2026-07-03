#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Single-item lot sizing problem implementations.

This module provides concrete implementations for single-item lot sizing variants:
- UncapacitatedSingleItemLSP: Wagner-Whitin problem (ULSP)
- CapacitatedSingleItemLSP: Single-item with capacity constraints
"""

from __future__ import annotations

import numpy as np

from discrete_optimization.lotsizing.backlog import (
    WithoutBacklogProblem,
    WithoutBacklogSolution,
)
from discrete_optimization.lotsizing.capacity import (
    WithoutCapacityProblem,
    WithoutCapacitySolution,
)
from discrete_optimization.lotsizing.changeover import (
    WithoutChangeoverCostsProblem,
    WithoutChangeoverCostsSolution,
)
from discrete_optimization.lotsizing.generic_lotsizing import (
    GenericLotSizingProblem,
)
from discrete_optimization.lotsizing.parallel_production import (
    WithoutParallelProductionProblem,
)
from discrete_optimization.lotsizing.production_solution import (
    ProductionBasedSolution,
    ProductionDecision,
)
from discrete_optimization.lotsizing.setup_times import (
    WithoutSetupTimesProblem,
)
from discrete_optimization.lotsizing.stock_limits import (
    WithoutStockLimitsProblem,
)


class UncapacitatedSingleItemLSP(
    WithoutCapacityProblem[int],
    WithoutBacklogProblem[int],
    WithoutChangeoverCostsProblem[int],
    WithoutSetupTimesProblem[int],
    WithoutParallelProductionProblem[int],
    WithoutStockLimitsProblem[int],
    GenericLotSizingProblem[int],
):
    """Uncapacitated single-item lot sizing problem (ULSP / Wagner-Whitin problem).

    This is the classical lot sizing problem:
    - Single product (item = 0)
    - No capacity constraints
    - No backlog allowed (demands must be satisfied on time)
    - No changeover costs (single item)
    - No setup times

    Can be solved optimally in O(T²) with Wagner-Whitin dynamic programming.

    The problem minimizes:
        sum_t (s_t * Y_t + v_t * X_t + c_t * I_t)

    Subject to:
        I_t = I_{t-1} + X_t - d_t    (inventory balance)
        X_t <= M * Y_t               (big-M constraint for setup)
        X_t, I_t >= 0                (non-negativity)
        Y_t in {0, 1}                (setup binary)
        I_0 = 0                      (no initial inventory)

    Where:
        s_t: setup cost in period t
        v_t: production cost per unit in period t
        c_t: inventory cost per unit in period t
        d_t: demand in period t
        X_t: production quantity in period t
        Y_t: setup indicator (1 if production, 0 otherwise)
        I_t: inventory at end of period t
    """

    def __init__(
        self,
        demands: list[int] | np.ndarray,
        setup_costs: list[float] | np.ndarray,
        production_costs: list[float] | np.ndarray,
        inventory_costs: list[float] | np.ndarray,
    ):
        """Initialize uncapacitated single-item lot sizing problem.

        Args:
            demands: Demand in each period (length T)
            setup_costs: Setup cost in each period (length T)
            production_costs: Variable production cost per unit in each period (length T)
            inventory_costs: Inventory holding cost per unit in each period (length T)
        """
        # Convert to numpy arrays
        self._demands = np.array(demands, dtype=np.int64)
        self._setup_costs = np.array(setup_costs, dtype=np.float64)
        self._production_costs = np.array(production_costs, dtype=np.float64)
        self._inventory_costs = np.array(inventory_costs, dtype=np.float64)

        # Validate dimensions
        T = len(self._demands)
        if len(self._setup_costs) != T:
            raise ValueError(
                f"setup_costs length {len(self._setup_costs)} != horizon {T}"
            )
        if len(self._production_costs) != T:
            raise ValueError(
                f"production_costs length {len(self._production_costs)} != horizon {T}"
            )
        if len(self._inventory_costs) != T:
            raise ValueError(
                f"inventory_costs length {len(self._inventory_costs)} != horizon {T}"
            )

        self._horizon = T
        self._items_list = [0]  # Single item with index 0

    @property
    def horizon(self) -> int:
        """Number of time periods."""
        return self._horizon

    @property
    def items_list(self) -> list[int]:
        """Single item with index 0."""
        return self._items_list

    # Demands mixin implementation
    def get_demand(self, item: int, period: int) -> int:
        """Get demand for item in period."""
        if item != 0:
            raise ValueError(f"Single item problem only has item 0, got {item}")
        return int(self._demands[period])

    # Costs mixin implementations
    def get_setup_cost(self, item: int, period: int) -> float:
        """Get setup cost."""
        if item != 0:
            raise ValueError(f"Single item problem only has item 0, got {item}")
        return float(self._setup_costs[period])

    def get_production_cost_per_unit(self, item: int, period: int) -> float:
        """Get production cost per unit."""
        if item != 0:
            raise ValueError(f"Single item problem only has item 0, got {item}")
        return float(self._production_costs[period])

    def get_inventory_cost_per_unit(self, item: int, period: int) -> float:
        """Get inventory cost per unit."""
        if item != 0:
            raise ValueError(f"Single item problem only has item 0, got {item}")
        return float(self._inventory_costs[period])

    def get_solution_type(self):
        """Return solution class for this problem."""
        return UncapacitatedSingleItemSolution


class UncapacitatedSingleItemSolution(
    ProductionBasedSolution[int],
    WithoutBacklogSolution[int],
    WithoutCapacitySolution[int],
    WithoutChangeoverCostsSolution[int],
):
    """Solution for uncapacitated single-item lot sizing problem.

    This uses ProductionBasedSolution which automatically computes:
    - Inventory levels
    - Delivery quantities
    - Backlog (0 for this problem)

    From the production decisions.
    """

    problem: UncapacitatedSingleItemLSP

    def __init__(
        self,
        problem: UncapacitatedSingleItemLSP,
        production_periods: list[int] | None = None,
        production_quantities: list[int] | None = None,
    ):
        """Initialize solution from production plan.

        Args:
            problem: Problem instance
            production_periods: Periods where production occurs (optional)
            production_quantities: Production quantities in each period (optional)

        If production_periods provided, production_quantities should match.
        Otherwise, provide production_quantities for all periods (0 means no production).
        """
        # Build production decisions list
        productions = []

        if production_periods is not None:
            # Sparse representation: only periods with production
            if production_quantities is None:
                raise ValueError(
                    "production_quantities required with production_periods"
                )
            if len(production_periods) != len(production_quantities):
                raise ValueError(
                    "production_periods and production_quantities must have same length"
                )

            for period, qty in zip(production_periods, production_quantities):
                if qty > 0:
                    productions.append(
                        ProductionDecision(item=0, period=period, quantity=qty)
                    )

        elif production_quantities is not None:
            # Dense representation: quantity for each period
            if len(production_quantities) != problem.horizon:
                raise ValueError(
                    f"production_quantities length must be {problem.horizon}"
                )

            for period, qty in enumerate(production_quantities):
                if qty > 0:
                    productions.append(
                        ProductionDecision(item=0, period=period, quantity=qty)
                    )
        else:
            # Empty solution (no production)
            pass

        # Initialize with ProductionBasedSolution (computes inventory/deliveries)
        ProductionBasedSolution.__init__(self, problem, productions)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"UncapacitatedSingleItemSolution("
            f"nb_setups={sum(1 for p in self.productions if p.quantity > 0)}, "
            f"total_cost={self.compute_total_cost():.2f})"
        )


def generate_random_instance(
    horizon: int,
    avg_demand: int = 10,
    setup_cost: float = 100.0,
    production_cost: float = 1.0,
    inventory_cost: float = 0.5,
    seed: int | None = None,
) -> UncapacitatedSingleItemLSP:
    """Generate random instance for testing.

    Args:
        horizon: Number of periods
        avg_demand: Average demand per period
        setup_cost: Fixed setup cost per period
        production_cost: Variable production cost per unit
        inventory_cost: Inventory holding cost per unit
        seed: Random seed for reproducibility

    Returns:
        Random problem instance
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate demands uniformly around average
    demands = np.random.randint(
        max(1, avg_demand - 5),
        avg_demand + 5,
        size=horizon,
    )

    # Constant costs for simplicity
    setup_costs = np.full(horizon, setup_cost)
    production_costs = np.full(horizon, production_cost)
    inventory_costs = np.full(horizon, inventory_cost)

    return UncapacitatedSingleItemLSP(
        demands=demands,
        setup_costs=setup_costs,
        production_costs=production_costs,
        inventory_costs=inventory_costs,
    )
