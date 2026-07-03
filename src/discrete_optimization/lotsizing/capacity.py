#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Capacity constraint mixins for lot sizing problems.

This module provides mixins for production capacity constraints, distinguishing
between uncapacitated and capacitated variants.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Generic

from discrete_optimization.lotsizing.base import Item
from discrete_optimization.lotsizing.demands import DemandsProblem, DemandsSolution

logger = logging.getLogger(__name__)


class CapacityProblem(DemandsProblem[Item], Generic[Item]):
    """Mixin for problems with production capacity constraints.

    Capacitated problems have a limit on total production time available in each period.
    The capacity constraint is typically:
        sum_i (p_it * X_it) <= h_t

    Where:
    - p_it: production time per unit of item i in period t
    - X_it: production quantity
    - h_t: available production time in period t
    """

    @abstractmethod
    def get_production_time_per_unit(self, item: Item, period: int) -> float:
        """Get production time per unit p_it.

        Args:
            item: Item identifier
            period: Time period

        Returns:
            Production time per unit (non-negative)
        """
        ...

    @abstractmethod
    def get_available_production_time(self, period: int) -> float:
        """Get available production time h_t in period t.

        Args:
            period: Time period

        Returns:
            Available capacity (non-negative, may be infinite for uncapacitated)
        """
        ...


class CapacitySolution(DemandsSolution[Item], Generic[Item]):
    """Solution mixin for capacity constraint checking."""

    problem: CapacityProblem[Item]

    def get_total_production_time_used(self, period: int) -> float:
        """Compute total production time used in period.

        This base implementation only considers production quantities.
        Subclasses (like SetupTimesSolution) may add setup times.

        Args:
            period: Time period

        Returns:
            Total production time used
        """
        total_time = 0.0
        for item in self.problem.items_list:
            qty = self.get_production_quantity(item, period)
            if qty > 0:
                time_per_unit = self.problem.get_production_time_per_unit(item, period)
                total_time += qty * time_per_unit
        return total_time

    def check_capacity_constraints(self) -> bool:
        """Check if capacity constraints are satisfied in all periods.

        Returns:
            True if capacity constraints satisfied, False otherwise
        """
        for t in range(self.problem.horizon):
            used = self.get_total_production_time_used(t)
            available = self.problem.get_available_production_time(t)

            # Use small tolerance for floating point comparison
            if used > available + 1e-6:
                logger.debug(
                    f"Capacity exceeded in period {t}: "
                    f"used {used:.2f} > available {available:.2f}"
                )
                return False
        return True

    def get_capacity_utilization(self, period: int) -> float:
        """Get capacity utilization ratio for a period.

        Args:
            period: Time period

        Returns:
            Ratio of used / available capacity (may be > 1 if violated)
        """
        used = self.get_total_production_time_used(period)
        available = self.problem.get_available_production_time(period)

        if available == 0:
            return float("inf") if used > 0 else 0.0

        return used / available


class WithoutCapacityProblem(CapacityProblem[Item], Generic[Item]):
    """Utility mixin for uncapacitated problems.

    Returns infinite capacity - no production time constraints.

    This is the "Without" variant following the generic_tasks_tools pattern.
    Use this when the problem is uncapacitated (ULSP - Uncapacitated Lot-Sizing Problem).
    """

    def get_production_time_per_unit(self, item: Item, period: int) -> float:
        """No time per unit constraint in uncapacitated problems."""
        return 0.0

    def get_available_production_time(self, period: int) -> float:
        """Infinite capacity available."""
        return float("inf")


class WithoutCapacitySolution(CapacitySolution[Item], Generic[Item]):
    """Solution mixin for uncapacitated problems.

    Capacity constraints are always satisfied (no constraints).
    """

    def check_capacity_constraints(self) -> bool:
        """Always satisfied for uncapacitated problems."""
        return True

    def get_capacity_utilization(self, period: int) -> float:
        """Utilization is undefined for uncapacitated problems."""
        return 0.0
