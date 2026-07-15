#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Parallel production constraint mixin for lot sizing problems.

This module provides mixins for problems where production of multiple items
in the same time period may or may not be allowed.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Generic

from discrete_optimization.lotsizing.base import Item
from discrete_optimization.lotsizing.demands import DemandsProblem, DemandsSolution

if TYPE_CHECKING:
    # Only import for type checking to avoid circular dependencies
    # and MRO conflicts
    pass

logger = logging.getLogger(__name__)


class ParallelProductionProblem(DemandsProblem[Item], Generic[Item]):
    """Mixin for problems with constraints on parallel production.

    Determines whether multiple items can be produced simultaneously in the same period.

    When parallel production is NOT allowed, the constraint is:
        sum_i Y_it <= 1  for all t

    Where Y_it is the binary setup variable indicating if item i is produced in period t.

    This models situations where:
    - Production line can only handle one product type at a time
    - Switching between items consumes the entire period
    - Production resources are exclusive (no multi-tasking)

    When parallel production IS allowed, multiple items can be produced in the same period.

    Relevant for multi-item problems. For single-item problems, this constraint
    is automatically satisfied.
    """

    @abstractmethod
    def allows_parallel_production(self) -> bool:
        """Check if multiple items can be produced in the same period.

        Returns:
            True if parallel production is allowed, False if only one item per period
        """
        ...


class ParallelProductionSolution(DemandsSolution[Item], Generic[Item]):
    """Solution mixin for parallel production constraint checking.

    Note: This mixin requires get_production_quantity(item, period) method to be available.
    In practice, this is provided by CapacitySolution or ProductionBasedSolution.
    Type checkers may warn about this - this is expected due to mixin composition.
    """

    problem: ParallelProductionProblem[Item]

    if TYPE_CHECKING:
        # Declare the method signature for type checking without enforcing inheritance
        def get_production_quantity(self, item: Item, period: int) -> int:
            """Get production quantity (provided by CapacitySolution)."""
            ...

    def get_items_produced_in_period(self, period: int) -> list[Item]:
        """Get list of items produced in a given period.

        Args:
            period: Time period

        Returns:
            List of items with positive production in this period
        """
        items_produced = []
        for item in self.problem.items_list:
            if self.get_production_quantity(item, period) > 0:
                items_produced.append(item)
        return items_produced

    def check_parallel_production_constraints(self) -> bool:
        """Check if parallel production constraints are satisfied.

        Returns:
            True if constraint satisfied, False otherwise
        """
        if self.problem.allows_parallel_production():
            # Constraint is not active - parallel production allowed
            return True

        # Check that at most one item is produced per period
        for t in range(self.problem.horizon):
            items_produced = self.get_items_produced_in_period(t)

            if len(items_produced) > 1:
                logger.debug(
                    f"Parallel production constraint violated in period {t}: "
                    f"{len(items_produced)} items produced {items_produced}. "
                    f"Only one item allowed per period."
                )
                return False

        return True

    def get_periods_with_parallel_production(self) -> list[tuple[int, list[Item]]]:
        """Get list of periods where multiple items are produced.

        Useful for identifying violations when parallel production is not allowed,
        or for analysis when it is allowed.

        Returns:
            List of (period, items_produced) tuples where len(items_produced) > 1
        """
        periods_with_parallel = []

        for t in range(self.problem.horizon):
            items_produced = self.get_items_produced_in_period(t)
            if len(items_produced) > 1:
                periods_with_parallel.append((t, items_produced))

        return periods_with_parallel

    def count_item_switches(self) -> int:
        """Count the number of periods where production switches to a different item.

        Useful for measuring setup frequency and production stability.

        Returns:
            Number of periods with item changes
        """
        switches = 0
        prev_item = None

        for t in range(self.problem.horizon):
            items = self.get_items_produced_in_period(t)
            curr_item = items[0] if len(items) == 1 else None

            if prev_item is not None and curr_item is not None:
                if prev_item != curr_item:
                    switches += 1

            if curr_item is not None:
                prev_item = curr_item

        return switches


class WithParallelProductionProblem(ParallelProductionProblem[Item], Generic[Item]):
    """Utility mixin for problems allowing parallel production.

    Multiple items can be produced simultaneously in the same period.

    This is the "With" variant for problems where parallel production
    of different items in the same period is allowed.
    """

    def allows_parallel_production(self) -> bool:
        """Parallel production is allowed."""
        return True


class WithParallelProductionSolution(ParallelProductionSolution[Item], Generic[Item]):
    """Solution mixin for problems allowing parallel production.

    The parallel production constraint is always satisfied (not active).
    """

    def check_parallel_production_constraints(self) -> bool:
        """Always satisfied when parallel production allowed."""
        return True


class WithoutParallelProductionProblem(ParallelProductionProblem[Item], Generic[Item]):
    """Utility mixin for problems NOT allowing parallel production.

    Only one item can be produced per period (exclusive production).

    This is the "Without" variant following the generic_tasks_tools pattern.
    """

    def allows_parallel_production(self) -> bool:
        """Parallel production is NOT allowed - only one item per period."""
        return False


class WithoutParallelProductionSolution(
    ParallelProductionSolution[Item], Generic[Item]
):
    """Solution mixin for problems NOT allowing parallel production.

    Provides full constraint checking for the single-item-per-period restriction.
    """

    # Inherits all checking methods from ParallelProductionSolution
    # check_parallel_production_constraints will enforce the constraint
    pass
