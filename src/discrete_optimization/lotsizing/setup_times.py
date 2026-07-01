#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Setup times mixin for lot sizing problems.

This module provides mixins for problems where setup operations consume
production capacity (in addition to production time).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Generic

from discrete_optimization.lotsizing.base import Item
from discrete_optimization.lotsizing.capacity import CapacityProblem, CapacitySolution


class SetupTimesProblem(CapacityProblem[Item], Generic[Item]):
    """Mixin for problems with setup times consuming capacity.

    Setup time τ_it is the time required to setup production for item i in period t.
    This time is added to the capacity constraint:
        sum_i (p_it * X_it + τ_it * Y_it) <= h_t

    Where Y_it = 1 if setup occurs (X_it > 0).
    """

    @abstractmethod
    def get_setup_time(self, item: Item, period: int) -> float:
        """Get setup time τ_it.

        Args:
            item: Item identifier
            period: Time period

        Returns:
            Setup time (non-negative)
        """
        ...


class SetupTimesSolution(CapacitySolution[Item], Generic[Item]):
    """Solution mixin for setup times in capacity constraints.

    This extends CapacitySolution to include setup times in the capacity calculation.
    """

    problem: SetupTimesProblem[Item]

    def get_total_production_time_used(self, period: int) -> float:
        """Override to include setup times in capacity usage.

        Total time = sum_i (p_it * X_it + τ_it * Y_it)

        Args:
            period: Time period

        Returns:
            Total production time including setup times
        """
        total_time = 0.0
        for item in self.problem.items_list:
            # Production time
            qty = self.get_production_quantity(item, period)
            if qty > 0:
                time_per_unit = self.problem.get_production_time_per_unit(item, period)
                total_time += qty * time_per_unit

            # Setup time (if setup occurs)
            if self.has_setup(item, period):
                setup_time = self.problem.get_setup_time(item, period)
                total_time += setup_time

        return total_time


class WithoutSetupTimesProblem(SetupTimesProblem[Item], Generic[Item]):
    """Utility mixin for problems without setup times.

    Setup times are zero - setups don't consume capacity.
    """

    def get_setup_time(self, item: Item, period: int) -> float:
        """No setup time."""
        return 0.0
