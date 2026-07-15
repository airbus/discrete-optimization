#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Cost structure mixins for lot sizing problems.

This module provides mixins for different cost components:
- Setup costs (fixed cost when production occurs)
- Production costs (variable cost per unit produced)
- Inventory costs (holding cost per unit in stock)
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Generic

import numpy as np

from discrete_optimization.lotsizing.base import Item
from discrete_optimization.lotsizing.demands import DemandsProblem, DemandsSolution


class SetupCostsProblem(DemandsProblem[Item], Generic[Item]):
    """Mixin for problems with setup costs.

    Setup cost s_it is the fixed cost incurred when producing item i in period t.
    This cost is paid if Y_it = 1 (setup occurs), regardless of production quantity.
    """

    @abstractmethod
    def get_setup_cost(self, item: Item, period: int) -> float:
        """Get setup cost for producing item i in period t.

        Args:
            item: Item identifier
            period: Time period

        Returns:
            Setup cost s_it (non-negative)
        """
        ...


class SetupCostsSolution(DemandsSolution[Item], Generic[Item]):
    """Solution mixin for setup cost computation."""

    problem: SetupCostsProblem[Item]

    @abstractmethod
    def has_setup(self, item: Item, period: int) -> bool:
        """Check if setup occurs (Y_it = 1).

        Args:
            item: Item identifier
            period: Time period

        Returns:
            True if setup occurs, False otherwise
        """
        ...

    def compute_total_setup_cost(self) -> float:
        """Compute total setup cost across all items and periods.

        Returns:
            Sum of all setup costs s_it * Y_it
        """
        total_cost = 0.0
        for item in self.problem.items_list:
            for t in range(self.problem.horizon):
                if self.has_setup(item, t):
                    total_cost += self.problem.get_setup_cost(item, t)
        return total_cost


class ProductionCostsProblem(DemandsProblem[Item], Generic[Item]):
    """Mixin for problems with variable production costs.

    Production cost v_it is the variable cost per unit of item i produced in period t.
    Total production cost = v_it * X_it where X_it is production quantity.
    """

    @abstractmethod
    def get_production_cost_per_unit(self, item: Item, period: int) -> float:
        """Get variable production cost per unit.

        Args:
            item: Item identifier
            period: Time period

        Returns:
            Production cost v_it per unit (non-negative)
        """
        ...


class ProductionCostsSolution(DemandsSolution[Item], Generic[Item]):
    """Solution mixin for production cost computation."""

    problem: ProductionCostsProblem[Item]

    def compute_total_production_cost(self) -> float:
        """Compute total variable production cost.

        Returns:
            Sum of v_it * X_it across all items and periods
        """
        total_cost = 0.0
        for item in self.problem.items_list:
            for t in range(self.problem.horizon):
                qty = self.get_production_quantity(item, t)
                if qty > 0:
                    cost_per_unit = self.problem.get_production_cost_per_unit(item, t)
                    total_cost += cost_per_unit * qty
        return total_cost


class InventoryCostsProblem(DemandsProblem[Item], Generic[Item]):
    """Mixin for problems with inventory holding costs.

    Inventory cost c_it is the cost per unit of item i held in stock at end of period t.
    Total inventory cost = c_it * I_it where I_it is inventory level.
    """

    @abstractmethod
    def get_inventory_cost_per_unit(self, item: Item, period: int) -> float:
        """Get inventory holding cost per unit.

        Args:
            item: Item identifier
            period: Time period

        Returns:
            Inventory cost c_it per unit (non-negative)
        """
        ...


class InventoryCostsSolution(DemandsSolution[Item], Generic[Item]):
    """Solution mixin for inventory cost computation."""

    problem: InventoryCostsProblem[Item]

    def compute_total_inventory_cost(self) -> float:
        """Compute total inventory holding cost.

        Returns:
            Sum of c_it * I_it across all items and periods
        """
        total_cost = 0.0
        for item in self.problem.items_list:
            for t in range(self.problem.horizon):
                inv = self.get_inventory_level(item, t)
                if inv > 0:
                    cost_per_unit = self.problem.get_inventory_cost_per_unit(item, t)
                    total_cost += cost_per_unit * inv
        return total_cost


# Concrete array-based implementations


class CostsArrayProblem(
    SetupCostsProblem[Item],
    ProductionCostsProblem[Item],
    InventoryCostsProblem[Item],
    Generic[Item],
):
    """Concrete implementation using numpy arrays for all cost components.

    This helper mixin stores costs as 2D arrays for efficient access.

    Can be used as:
        class MyProblem(CostsArrayProblem[int], OtherMixins...):
            def __init__(self, setup_costs, production_costs, inventory_costs, ...):
                CostsArrayProblem.__init__(
                    self, setup_costs, production_costs, inventory_costs
                )
                ...
    """

    def __init__(
        self,
        setup_costs: np.ndarray | list[list[float]],
        production_costs: np.ndarray | list[list[float]],
        inventory_costs: np.ndarray | list[list[float]],
    ):
        """Initialize with cost arrays.

        Args:
            setup_costs: 2D array of setup_costs[item_idx][period]
            production_costs: 2D array of production_costs[item_idx][period]
            inventory_costs: 2D array of inventory_costs[item_idx][period]
        """
        self._setup_costs = (
            np.array(setup_costs, dtype=np.float64)
            if not isinstance(setup_costs, np.ndarray)
            else setup_costs
        )
        self._production_costs = (
            np.array(production_costs, dtype=np.float64)
            if not isinstance(production_costs, np.ndarray)
            else production_costs
        )
        self._inventory_costs = (
            np.array(inventory_costs, dtype=np.float64)
            if not isinstance(inventory_costs, np.ndarray)
            else inventory_costs
        )

    def get_setup_cost(self, item: Item, period: int) -> float:
        """Get setup cost from array storage."""
        idx = self.get_index_from_item(item)
        return float(self._setup_costs[idx, period])

    def get_production_cost_per_unit(self, item: Item, period: int) -> float:
        """Get production cost per unit from array storage."""
        idx = self.get_index_from_item(item)
        return float(self._production_costs[idx, period])

    def get_inventory_cost_per_unit(self, item: Item, period: int) -> float:
        """Get inventory cost per unit from array storage."""
        idx = self.get_index_from_item(item)
        return float(self._inventory_costs[idx, period])


class SingleItemCostsArrayProblem(
    SetupCostsProblem[int],
    ProductionCostsProblem[int],
    InventoryCostsProblem[int],
):
    """Concrete implementation for single-item problems with 1D cost arrays.

    Convenience class for single-item problems where costs are stored as 1D arrays.
    The items_list is fixed to [0].
    """

    def __init__(
        self,
        setup_costs: np.ndarray | list[float],
        production_costs: np.ndarray | list[float],
        inventory_costs: np.ndarray | list[float],
    ):
        """Initialize with 1D cost arrays.

        Args:
            setup_costs: 1D array of setup_costs[period]
            production_costs: 1D array of production_costs[period]
            inventory_costs: 1D array of inventory_costs[period]
        """
        self._setup_costs = (
            np.array(setup_costs, dtype=np.float64)
            if not isinstance(setup_costs, np.ndarray)
            else setup_costs
        )
        self._production_costs = (
            np.array(production_costs, dtype=np.float64)
            if not isinstance(production_costs, np.ndarray)
            else production_costs
        )
        self._inventory_costs = (
            np.array(inventory_costs, dtype=np.float64)
            if not isinstance(inventory_costs, np.ndarray)
            else inventory_costs
        )

    @property
    def items_list(self) -> list[int]:
        """Single item with index 0."""
        return [0]

    @property
    def horizon(self) -> int:
        """Horizon is length of cost arrays."""
        return len(self._setup_costs)

    def get_setup_cost(self, item: int, period: int) -> float:
        """Get setup cost from 1D array."""
        if item != 0:
            raise ValueError(f"Single item problem only has item 0, got {item}")
        return float(self._setup_costs[period])

    def get_production_cost_per_unit(self, item: int, period: int) -> float:
        """Get production cost per unit from 1D array."""
        if item != 0:
            raise ValueError(f"Single item problem only has item 0, got {item}")
        return float(self._production_costs[period])

    def get_inventory_cost_per_unit(self, item: int, period: int) -> float:
        """Get inventory cost per unit from 1D array."""
        if item != 0:
            raise ValueError(f"Single item problem only has item 0, got {item}")
        return float(self._inventory_costs[period])


# "Without" variants for problems without specific cost components


class WithoutSetupCostsProblem(SetupCostsProblem[Item], Generic[Item]):
    """Mixin for problems without setup costs.

    Use this when there is no fixed cost to start production.
    All setup costs return 0.
    """

    def get_setup_cost(self, item: Item, period: int) -> float:
        """No setup costs - always returns 0."""
        return 0.0


class WithoutSetupCostsSolution(SetupCostsSolution[Item], Generic[Item]):
    """Solution mixin for problems without setup costs."""

    problem: WithoutSetupCostsProblem[Item]

    def compute_total_setup_cost(self) -> float:
        """No setup costs - always returns 0."""
        return 0.0


class WithoutProductionCostsProblem(ProductionCostsProblem[Item], Generic[Item]):
    """Mixin for problems without per-unit production costs.

    Use this when production is limited only by capacity, not by per-unit costs.
    All production costs return 0.
    """

    def get_production_cost_per_unit(self, item: Item, period: int) -> float:
        """No production costs - always returns 0."""
        return 0.0


class WithoutProductionCostsSolution(ProductionCostsSolution[Item], Generic[Item]):
    """Solution mixin for problems without production costs."""

    problem: WithoutProductionCostsProblem[Item]

    def compute_total_production_cost(self) -> float:
        """No production costs - always returns 0."""
        return 0.0


class WithoutInventoryCostsProblem(InventoryCostsProblem[Item], Generic[Item]):
    """Mixin for problems without inventory holding costs.

    Use this when inventory can be held without cost (rare in practice).
    All inventory costs return 0.
    """

    def get_inventory_cost_per_unit(self, item: Item, period: int) -> float:
        """No inventory costs - always returns 0."""
        return 0.0


class WithoutInventoryCostsSolution(InventoryCostsSolution[Item], Generic[Item]):
    """Solution mixin for problems without inventory costs."""

    problem: WithoutInventoryCostsProblem[Item]

    def compute_total_inventory_cost(self) -> float:
        """No inventory costs - always returns 0."""
        return 0.0
