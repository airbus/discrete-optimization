#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Demands mixin for lot sizing problems.

This module provides the core demands component that nearly all lot sizing problems use.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Generic

import numpy as np

from discrete_optimization.lotsizing.base import (
    Item,
    LotSizingProblem,
    LotSizingSolution,
)

logger = logging.getLogger(__name__)


class DemandsProblem(LotSizingProblem[Item], Generic[Item]):
    """Mixin for problems with demand requirements.

    This is a core component - nearly all lot sizing problems have demands to satisfy.

    The demand d_it represents the quantity of item i required in period t.
    """

    @abstractmethod
    def get_demand(self, item: Item, period: int) -> int:
        """Get demand for given item in given period.

        Args:
            item: The product/item type
            period: Time period (0 to horizon-1)

        Returns:
            Demand quantity d_it (non-negative integer)
        """
        ...

    def get_total_demand(self, item: Item) -> int:
        """Get total demand for an item across all periods.

        Returns:
            Sum of all demands for this item
        """
        return sum(self.get_demand(item, t) for t in range(self.horizon))

    def is_binary_demand_item(self, item: Item) -> bool:
        return all(
            int(self.get_demand(item, period=t)) in {0, 1} for t in range(self.horizon)
        )

    def is_binary_demand(self):
        return all(self.is_binary_demand_item(item=item) for item in self.items_list)

    def get_cumulative_demands(self, item: Item) -> np.ndarray:
        """Get cumulative demand for item over time.

        Useful for inventory and delivery computations.

        Returns:
            Array of cumulative demands [d_i0, d_i0+d_i1, d_i0+d_i1+d_i2, ...]
        """
        demands = [self.get_demand(item, t) for t in range(self.horizon)]
        return np.cumsum(demands)

    def get_max_demand_per_period(self) -> int:
        """Get maximum demand across all items and periods.

        Useful for setting upper bounds in solvers.

        Returns:
            max_i,t d_it
        """
        return max(
            self.get_demand(item, t)
            for item in self.items_list
            for t in range(self.horizon)
        )

    @abstractmethod
    def allows_lost_demand(self) -> bool: ...


class DemandsSolution(LotSizingSolution[Item], Generic[Item]):
    """Solution mixin for demand-based problems.

    Provides methods to check demand satisfaction.
    """

    problem: DemandsProblem[Item]

    def check_demand_satisfaction(self, allow_delays: bool = False) -> bool:
        """Check whether all demands are eventually satisfied.

        Args:
            allow_delays: If False, demands must be satisfied on time (no backlog).
                          If True, backlog is allowed but total satisfaction required.

        Returns:
            True if demands are satisfied according to policy, False otherwise
        """
        if not self.problem.allows_lost_demand():
            for item in self.problem.items_list:
                total_delivery = self.get_total_delivery_quantity(item)
                if total_delivery < self.problem.get_total_demand(item):
                    logger.debug(
                        f"Some demands not satisfied: {item}, "
                        f"{total_delivery} vs {self.problem.get_total_demand(item)}"
                    )
                    return False
        for item in self.problem.items_list:
            cumul_delivery = 0
            cumul_demand = 0

            for t in range(self.problem.horizon):
                cumul_delivery += self.get_delivery_quantity(item, t)
                cumul_demand += self.problem.get_demand(item, t)

                if not allow_delays and cumul_delivery < cumul_demand:
                    logger.debug(
                        f"Demand not satisfied on time for item {item} at period {t}: "
                        f"cumulative delivered {cumul_delivery} < cumulative demand {cumul_demand}"
                    )
                    return False

            # Check total satisfaction at end of horizon
            if cumul_delivery < cumul_demand:
                logger.debug(
                    f"Total demand not satisfied for item {item}: "
                    f"total delivered {cumul_delivery} < total demand {cumul_demand}"
                )
                return False
        return True

    def get_total_unmet_demand(self) -> int:
        """Compute total unmet demand across all items and periods.

        Returns:
            Total quantity of demand not satisfied
        """
        total_unmet = 0
        for item in self.problem.items_list:
            cumul_delivery = sum(
                self.get_delivery_quantity(item, t) for t in range(self.problem.horizon)
            )
            cumul_demand = self.problem.get_total_demand(item)
            total_unmet += max(0, cumul_demand - cumul_delivery)
        return total_unmet


class DemandsArrayProblem(DemandsProblem[Item], Generic[Item]):
    """Concrete implementation of DemandsProblem using numpy arrays for storage.

    This is a helper mixin for concrete problem classes that want to store
    demands as a 2D array.

    Can be used as:
        class MyProblem(DemandsArrayProblem[int], OtherMixins...):
            def __init__(self, demands, ...):
                DemandsArrayProblem.__init__(self, demands)
                ...
    """

    def __init__(self, demands: np.ndarray | list[list[int]]):
        """Initialize with demands array.

        Args:
            demands: 2D array of shape (nb_items, horizon) or list of lists
                    demands[i][t] = demand for item i in period t
        """
        if not isinstance(demands, np.ndarray):
            demands = np.array(demands, dtype=np.int64)
        self._demands = demands

    def get_demand(self, item: Item, period: int) -> int:
        """Get demand from array storage."""
        item_idx = self.get_index_from_item(item)
        return int(self._demands[item_idx, period])


class SingleItemDemandsArrayProblem(DemandsProblem[int]):
    """Concrete implementation for single-item problems with array storage.

    This is a convenience class for single-item problems where demands
    can be stored as a 1D array.

    The items_list is fixed to [0].
    """

    def __init__(self, demands: np.ndarray | list[int]):
        """Initialize with 1D demands array.

        Args:
            demands: 1D array of demands[t] for each period t
        """
        if not isinstance(demands, np.ndarray):
            demands = np.array(demands, dtype=np.int64)
        self._demands = demands

    @property
    def items_list(self) -> list[int]:
        """Single item with index 0."""
        return [0]

    @property
    def horizon(self) -> int:
        """Horizon is length of demands array."""
        return len(self._demands)

    def get_demand(self, item: int, period: int) -> int:
        """Get demand from 1D array."""
        if item != 0:
            raise ValueError(f"Single item problem only has item 0, got {item}")
        return int(self._demands[period])
