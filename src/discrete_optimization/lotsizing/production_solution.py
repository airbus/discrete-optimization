#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Generic production-based solution with inventory and backlog computation.

This module provides a base solution class that handles the core logic of computing
inventory levels, deliveries, and backlog from production decisions. This should work
for most lot sizing variants and provides a solid foundation for the mixin solutions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from discrete_optimization.lotsizing.base import Item, LotSizingProblem
from discrete_optimization.lotsizing.generic_lotsizing import (
    GenericLotSizingProblem,
    GenericLotSizingSolution,
)

logger = logging.getLogger(__name__)


@dataclass
class ProductionDecision:
    """Represents a production decision.

    Attributes:
        item: Item/product type being produced
        period: Time period of production (0 to horizon-1)
        quantity: Production quantity X_it
        setup: Whether a setup Y_it occurs (derived from quantity > 0)
    """

    item: int  # Using int for simplicity, will be generic in solutions
    period: int
    quantity: int

    @property
    def setup(self) -> bool:
        """Setup occurs if production quantity > 0."""
        return self.quantity > 0


@dataclass
class DeliveryDecision:
    """Represents a delivery decision.

    Attributes:
        item: Item/product type being delivered
        period: Time period of delivery (0 to horizon-1)
        quantity: Delivery quantity D_it
    """

    item: int  # Using int for simplicity, will be generic in solutions
    period: int
    quantity: int


class ProductionBasedSolution(GenericLotSizingSolution[Item]):
    """Generic solution based on production decisions.

    This class provides a concrete implementation of GenericLotSizingSolution
    that automatically computes inventory, deliveries, and backlog from production decisions.

    Key features:
    - Inventory levels computed over time
    - Delivery quantities to satisfy demands
    - Backlog quantities (delayed demands)

    The computation follows the inventory balance equation:
        I_it = I_i,t-1 + X_it - D_it

    Where:
    - I_it: Inventory at end of period t
    - X_it: Production in period t
    - D_it: Delivery in period t (satisfying demand)

    This implementation assumes:
    - Productions are provided as list of ProductionDecision objects
    - Demands are available via problem.get_demand() (from DemandsProblem mixin)
    - Deliveries are computed to satisfy demands ASAP from available stock

    Subclasses can override delivery computation for different policies.
    Subclasses automatically get all GenericLotSizingSolution mixin methods
    (check_demand_satisfaction, check_capacity_constraints, compute_total_*_cost, etc.)
    """

    problem: GenericLotSizingProblem[Item]

    def __init__(
        self,
        problem: LotSizingProblem[Item],
        productions: list[ProductionDecision],
        deliveries: list[DeliveryDecision] | None = None,
    ):
        """Initialize production-based solution.

        Args:
            problem: The lot sizing problem instance
            productions: List of production decisions
            deliveries: Optional list of delivery decisions. If provided, these will be used
                       directly instead of being computed from production and demand.
                       If None, deliveries will be computed automatically.
        """
        super().__init__(problem)
        self.productions = productions
        self.deliveries = deliveries

        # Computed derived values (cached)
        self._inventory_levels: dict[Item, np.ndarray] | None = None
        self._delivery_quantities: dict[Item, np.ndarray] | None = None
        self._backlog_quantities: dict[Item, np.ndarray] | None = None
        self._production_array: dict[Item, np.ndarray] | None = None

        # Compute all derived values
        self._compute_all_derived_values()

    def _compute_all_derived_values(self) -> None:
        """Compute inventory, deliveries, and backlog from production decisions.

        This implements the core lot sizing logic:
        1. Build production array from decisions
        2. Build delivery array from decisions (if provided) or compute from stock
        3. Compute inventory levels (stock - deliveries)
        4. Compute backlog (unsatisfied cumulative demand)
        """
        # Reset cached values
        self._inventory_levels = {}
        self._delivery_quantities = {}
        self._backlog_quantities = {}
        self._production_array = {}

        # Build production array: production[item][period] = quantity
        production_dict: dict[Item, dict[int, int]] = {
            item: {t: 0 for t in range(self.problem.horizon)}
            for item in self.problem.items_list
        }

        for prod in self.productions:
            item = (
                self.problem.get_item_from_index(prod.item)
                if isinstance(prod.item, int)
                else prod.item
            )
            if item in production_dict:
                production_dict[item][prod.period] = prod.quantity

        # Convert to numpy arrays for efficient computation
        for item in self.problem.items_list:
            prod_array = np.array(
                [production_dict[item][t] for t in range(self.problem.horizon)],
                dtype=np.int64,
            )
            self._production_array[item] = prod_array

        # Build delivery array if deliveries are provided
        if self.deliveries is not None:
            delivery_dict: dict[Item, dict[int, int]] = {
                item: {t: 0 for t in range(self.problem.horizon)}
                for item in self.problem.items_list
            }

            for deliv in self.deliveries:
                item = (
                    self.problem.get_item_from_index(deliv.item)
                    if isinstance(deliv.item, int)
                    else deliv.item
                )
                if item in delivery_dict:
                    delivery_dict[item][deliv.period] = deliv.quantity

            # Convert to numpy arrays
            for item in self.problem.items_list:
                deliv_array = np.array(
                    [delivery_dict[item][t] for t in range(self.problem.horizon)],
                    dtype=np.int64,
                )
                self._delivery_quantities[item] = deliv_array

        # Compute for each item
        for item in self.problem.items_list:
            self._compute_item_inventory_and_deliveries(item)

    def _compute_item_inventory_and_deliveries(self, item: Item) -> None:
        """Compute inventory, deliveries, and backlog for a single item.

        If deliveries were provided in __init__, use them directly.
        Otherwise, implement a greedy delivery policy: deliver as much as possible
        from available stock to satisfy cumulative demand.

        Args:
            item: Item to compute for
        """
        horizon = self.problem.horizon

        # Get production array
        production = self._production_array[item]

        # Get demands if problem has DemandsProblem mixin
        try:
            demands = np.array(
                [self.problem.get_demand(item, t) for t in range(horizon)],
                dtype=np.int64,
            )
        except AttributeError:
            # No demands defined, assume zero
            demands = np.zeros(horizon, dtype=np.int64)

        # Initialize arrays
        inventory = np.zeros(horizon, dtype=np.int64)
        backlog = np.zeros(horizon, dtype=np.int64)

        # Check if deliveries were provided or need to be computed
        if item in self._delivery_quantities:
            # Deliveries already set from provided list
            deliveries = self._delivery_quantities[item]
        else:
            # Need to compute deliveries
            deliveries = np.zeros(horizon, dtype=np.int64)

        # Track cumulative quantities
        cumul_production = 0
        cumul_demand = 0
        cumul_delivered = 0

        for t in range(horizon):
            # Update cumulative production
            cumul_production += int(production[t])

            # Update cumulative demand
            cumul_demand += int(demands[t])

            if item not in self._delivery_quantities:
                # Compute deliveries if not provided
                # Compute stock available (production so far - delivered so far)
                stock_available = cumul_production - cumul_delivered

                # Compute how much we can deliver (limited by stock and remaining demand)
                remaining_demand = cumul_demand - cumul_delivered
                can_deliver = min(stock_available, remaining_demand)

                deliveries[t] = can_deliver

            cumul_delivered += int(deliveries[t])

            # Update inventory (stock remaining after delivery)
            inventory[t] = cumul_production - cumul_delivered

            # Update backlog (cumulative demand not yet satisfied)
            backlog[t] = cumul_demand - cumul_delivered

        # Store computed values
        self._inventory_levels[item] = inventory
        if item not in self._delivery_quantities:
            self._delivery_quantities[item] = deliveries
        self._backlog_quantities[item] = backlog

    def invalidate_cache(self) -> None:
        """Invalidate cached computed values.

        Call this when productions are modified externally.
        """
        self._inventory_levels = None
        self._delivery_quantities = None
        self._backlog_quantities = None
        self._production_array = None

    def get_production_quantity(self, item: Item, period: int) -> int:
        """Get production quantity for given item and period.

        Args:
            item: Item identifier
            period: Time period

        Returns:
            Production quantity X_it
        """
        if self._production_array is None:
            self._compute_all_derived_values()
        return int(self._production_array[item][period])

    def has_setup(self, item: Item, period: int) -> bool:
        """Check if setup occurs for given item and period.

        Setup occurs if production quantity > 0.

        Args:
            item: Item identifier
            period: Time period

        Returns:
            True if setup Y_it = 1, False otherwise
        """
        return self.get_production_quantity(item, period) > 0

    def get_delivery_quantity(self, item: Item, period: int) -> int:
        """Get delivery quantity for given item and period.

        Delivery quantity D_it is the amount delivered to satisfy demand in period t.

        Args:
            item: Item identifier
            period: Time period

        Returns:
            Delivery quantity D_it
        """
        if self._delivery_quantities is None:
            self._compute_all_derived_values()
        return int(self._delivery_quantities[item][period])

    def get_inventory_level(self, item: Item, period: int) -> int:
        """Get inventory level at end of period.

        Inventory I_it is the stock remaining at end of period t.

        Args:
            item: Item identifier
            period: Time period

        Returns:
            Inventory level I_it
        """
        if self._inventory_levels is None:
            self._compute_all_derived_values()
        return int(self._inventory_levels[item][period])

    def get_backlog_quantity(self, item: Item, period: int) -> int:
        """Get backlog quantity at end of period.

        Backlog B_it is the cumulative demand not yet satisfied at end of period t.

        Args:
            item: Item identifier
            period: Time period

        Returns:
            Backlog quantity B_it
        """
        if self._backlog_quantities is None:
            self._compute_all_derived_values()
        return int(self._backlog_quantities[item][period])

    def get_production_sequence(self) -> list[tuple[int, Item]]:
        """Get production sequence as list of (period, item) tuples.

        Sorted by period, useful for computing changeover costs.

        Returns:
            List of (period, item) tuples where production occurs
        """
        sequence = []
        for prod in self.productions:
            if prod.quantity > 0:  # Only include actual production (setup)
                item = (
                    self.problem.get_item_from_index(prod.item)
                    if isinstance(prod.item, int)
                    else prod.item
                )
                sequence.append((prod.period, item))

        # Sort by period
        sequence.sort(key=lambda x: x[0])
        return sequence

    def get_production_quantity_array(self, item: Item) -> np.ndarray:
        """Get production quantities for all periods for given item.

        Args:
            item: Item identifier

        Returns:
            Array of production quantities [X_i0, X_i1, ..., X_i,T-1]
        """
        if self._production_array is None:
            self._compute_all_derived_values()
        return self._production_array[item].copy()

    def get_delivery_quantity_array(self, item: Item) -> np.ndarray:
        """Get delivery quantities for all periods for given item.

        Args:
            item: Item identifier

        Returns:
            Array of delivery quantities [D_i0, D_i1, ..., D_i,T-1]
        """
        if self._delivery_quantities is None:
            self._compute_all_derived_values()
        return self._delivery_quantities[item].copy()

    def get_inventory_level_array(self, item: Item) -> np.ndarray:
        """Get inventory levels for all periods for given item.

        Args:
            item: Item identifier

        Returns:
            Array of inventory levels [I_i0, I_i1, ..., I_i,T-1]
        """
        if self._inventory_levels is None:
            self._compute_all_derived_values()
        return self._inventory_levels[item].copy()

    def get_backlog_quantity_array(self, item: Item) -> np.ndarray:
        """Get backlog quantities for all periods for given item.

        Args:
            item: Item identifier

        Returns:
            Array of backlog quantities [B_i0, B_i1, ..., B_i,T-1]
        """
        if self._backlog_quantities is None:
            self._compute_all_derived_values()
        return self._backlog_quantities[item].copy()

    def copy(self) -> ProductionBasedSolution:
        """Create a copy of this solution.

        Returns:
            New solution with copied production and delivery decisions
        """
        return ProductionBasedSolution(
            problem=self.problem,
            productions=list(self.productions),
            deliveries=list(self.deliveries) if self.deliveries is not None else None,
        )

    def lazy_copy(self) -> ProductionBasedSolution:
        """Create a lazy copy sharing production and delivery lists.

        Warning: Modifying productions or deliveries will affect both solutions.

        Returns:
            New solution sharing production and delivery lists
        """
        return ProductionBasedSolution(
            problem=self.problem,
            productions=self.productions,
            deliveries=self.deliveries,
        )

    def __repr__(self) -> str:
        """String representation of solution."""
        return (
            f"ProductionBasedSolution("
            f"nb_productions={len(self.productions)}, "
            f"horizon={self.problem.horizon}, "
            f"nb_items={self.problem.nb_items})"
        )
