#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Stock limits mixin for lot sizing problems.

This module provides mixins for problems with inventory stock limits.
Stock limits constrain the maximum inventory that can be held for each item
in each period.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Generic

from pandas.core.indexes import period

from discrete_optimization.lotsizing.base import Item
from discrete_optimization.lotsizing.demands import DemandsProblem, DemandsSolution

logger = logging.getLogger(__name__)


class StockLimitsProblem(DemandsProblem[Item], Generic[Item]):
    """Mixin for problems with inventory stock limits.

    Stock limits S_it constrain the maximum inventory that can be held:
        I_it <= S_it

    Where:
    - I_it: inventory level for item i at end of period t
    - S_it: maximum allowed stock for item i in period t

    This can model warehouse capacity constraints, perishability limits,
    or other storage restrictions.
    """

    @abstractmethod
    def get_stock_limit_for_item(self, item: Item, period: int) -> int | float:
        """Get maximum inventory/stock limit for item in period.

        Args:
            item: Item identifier
            period: Time period

        Returns:
            Maximum allowed inventory S_it (non-negative, may be infinite)
        """
        ...

    @abstractmethod
    def get_overall_stock_limit(self, period: int) -> int | float: ...

    def has_overall_stock_limit(self, period: int) -> bool:
        val = self.get_overall_stock_limit(period)
        if val is not None and val != float("inf"):
            return True
        else:
            return False

    def has_stock_limit_for_item(self, item: Item, period: int) -> bool:
        val = self.get_stock_limit_for_item(item, period)
        if val is not None and val != float("inf"):
            return True
        else:
            return False

    def has_stock_limits(self) -> bool:
        """Check if stock limits are active (any limit is finite).

        Returns:
            True if any stock limit is not infinite
        """
        for t in range(self.horizon):
            if self.has_overall_stock_limit(period):
                return True
            for item in self.items_list:
                if self.has_stock_limit_for_item(item=item, period=t):
                    return True
        return False


class StockLimitsSolution(DemandsSolution[Item], Generic[Item]):
    """Solution mixin for stock limit constraint checking.

    Note: This mixin requires get_inventory_level(item, period) method to be available.
    In practice, this is provided by InventoryCostsSolution or ProductionBasedSolution.
    Type checkers may warn about this - this is expected due to mixin composition.
    """

    problem: StockLimitsProblem[Item]

    def check_stock_limit_constraints(self) -> bool:
        """Check if stock limits are satisfied in all periods.

        Returns:
            True if all stock limits satisfied, False otherwise
        """
        for t in range(self.problem.horizon):
            overall_inventory = 0
            overall_limit = self.problem.get_overall_stock_limit(period=t)
            for item in self.problem.items_list:
                inventory = self.get_inventory_level(item, t)
                limit = self.problem.get_stock_limit_for_item(item, t)
                overall_inventory += inventory
                # Use small tolerance for floating point comparison
                if inventory > limit + 1e-6:
                    logger.debug(
                        f"Stock limit exceeded for item {item} in period {t}: "
                        f"inventory {inventory} > limit {limit}"
                    )
                    return False
                if inventory < 0:
                    logger.debug(f"Negative inventory for item {item} in period {t}")
                    return False
                if overall_limit > overall_limit + 1e-6:
                    logger.debug(
                        f"Overall stock limit exceeded in period {t}: "
                        f"inventory {overall_inventory} > limit {overall_limit}"
                    )
                    return False

        return True

    def get_stock_limit_violations(self) -> list[tuple[Item, int, float]]:
        """Get list of stock limit violations.

        Returns:
            List of (item, period, excess) tuples where excess = inventory - limit
        """
        violations = []
        for item in self.problem.items_list:
            for t in range(self.problem.horizon):
                inventory = self.get_inventory_level(item, t)
                limit = self.problem.get_stock_limit_for_item(item, t)

                excess = inventory - limit
                if excess > 1e-6:
                    violations.append((item, t, excess))

        return violations

    def get_max_stock_utilization(self) -> float:
        """Get maximum stock utilization ratio across all items and periods.

        Returns:
            max_i,t (I_it / S_it), or 0 if no limits exist
        """
        max_util = 0.0
        for item in self.problem.items_list:
            for t in range(self.problem.horizon):
                limit = self.problem.get_stock_limit_for_item(item, t)
                if limit > 0 and limit != float("inf"):
                    inventory = self.get_inventory_level(item, t)
                    utilization = inventory / limit
                    max_util = max(max_util, utilization)

        return max_util


class WithoutStockLimitsProblem(StockLimitsProblem[Item], Generic[Item]):
    """Utility mixin for problems without stock limits.

    Returns infinite limits - no inventory constraints.

    This is the "Without" variant following the generic_tasks_tools pattern.
    Use this when there are no warehouse capacity or storage constraints.
    """

    def get_stock_limit_for_item(self, item: Item, period: int) -> float:
        """Infinite stock limit - no constraint."""
        return float("inf")

    def get_overall_stock_limit(self, period: int) -> int | float:
        return float("inf")

    def has_stock_limits(self) -> bool:
        """No stock limits."""
        return False


class WithoutStockLimitsSolution(StockLimitsSolution[Item], Generic[Item]):
    """Solution mixin for problems without stock limits.

    Stock limit constraints are always satisfied (no constraints).
    """

    def check_stock_limit_constraints(self) -> bool:
        """Always satisfied for unlimited stock."""
        return True

    def get_stock_limit_violations(self) -> list[tuple[Item, int, float]]:
        """No violations when unlimited."""
        return []

    def get_max_stock_utilization(self) -> float:
        """Utilization is undefined for unlimited stock."""
        return 0.0
