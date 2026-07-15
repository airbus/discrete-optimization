#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Backlog feature mixin for lot sizing problems.

This module provides mixins for problems that allow backlogged demand
(demand satisfied in later periods).
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Generic

from discrete_optimization.lotsizing.base import Item
from discrete_optimization.lotsizing.demands import DemandsProblem, DemandsSolution

logger = logging.getLogger(__name__)


class BacklogProblem(DemandsProblem[Item], Generic[Item]):
    """Mixin for problems allowing backlogged demand.

    Backlog B_it represents the cumulative demand not yet satisfied at end of period t.
    A cost b_it is incurred per unit of backlog.

    When backlog is allowed, the demand satisfaction constraint is relaxed:
    instead of requiring delivery in period t, demand can be satisfied in later periods.
    """

    @abstractmethod
    def get_backlog_cost_per_unit(self, item: Item, period: int) -> float:
        """Get cost per unit of backlogged demand.

        Args:
            item: Item identifier
            period: Time period

        Returns:
            Backlog cost b_it per unit (non-negative)
        """
        ...

    @abstractmethod
    def is_backlog_allowed(self) -> bool:
        """Check whether backlog is allowed in this problem.

        Returns:
            True if backlog is permitted, False if demands must be satisfied on time
        """
        ...


class BacklogSolution(DemandsSolution[Item], Generic[Item]):
    """Solution mixin for backlog handling."""

    problem: BacklogProblem[Item]

    @abstractmethod
    def get_backlog_quantity(self, item: Item, period: int) -> int:
        """Get backlog quantity at end of period.

        Backlog is the cumulative demand not yet satisfied:
            B_it = max(0, cumulative_demand_it - cumulative_delivery_it)

        Args:
            item: Item identifier
            period: Time period

        Returns:
            Backlog quantity B_it (non-negative integer)
        """
        ...

    def compute_total_backlog_cost(self) -> float:
        """Compute total backlog cost across all items and periods.

        Returns:
            Sum of b_it * B_it
        """
        total_cost = 0.0
        for item in self.problem.items_list:
            for t in range(self.problem.horizon):
                backlog = self.get_backlog_quantity(item, t)
                if backlog > 0:
                    cost_per_unit = self.problem.get_backlog_cost_per_unit(item, t)
                    total_cost += cost_per_unit * backlog
        return total_cost

    def check_backlog_constraints(self) -> bool:
        """Check backlog constraints.

        If backlog is not allowed, verify that no backlog exists (all demands satisfied on time).

        Returns:
            True if constraints satisfied, False otherwise
        """
        if not self.problem.is_backlog_allowed():
            # Backlog not allowed - check that all demands satisfied on time
            for item in self.problem.items_list:
                for t in range(self.problem.horizon):
                    backlog = self.get_backlog_quantity(item, t)
                    if backlog > 0:
                        logger.debug(
                            f"Backlog not allowed but found {backlog} units "
                            f"for item {item} at period {t}"
                        )
                        return False
        return True

    def get_max_backlog(self) -> int:
        """Get maximum backlog across all items and periods.

        Useful for solution quality assessment.

        Returns:
            max_i,t B_it
        """
        return max(
            self.get_backlog_quantity(item, t)
            for item in self.problem.items_list
            for t in range(self.problem.horizon)
        )

    def get_total_backlog_at_period(self, period: int) -> int:
        """Get total backlog across all items at a given period.

        Args:
            period: Time period

        Returns:
            Sum of backlog for all items at this period
        """
        return sum(
            self.get_backlog_quantity(item, period) for item in self.problem.items_list
        )


class WithoutBacklogProblem(BacklogProblem[Item], Generic[Item]):
    """Utility mixin for problems without backlog.

    This is the "Without" variant for problems where demands must be satisfied on time.
    Backlog costs are zero and backlog is not allowed.
    """

    def get_backlog_cost_per_unit(self, item: Item, period: int) -> float:
        """No backlog cost."""
        return 0.0

    def is_backlog_allowed(self) -> bool:
        """Backlog not allowed."""
        return False


class WithoutBacklogSolution(BacklogSolution[Item], Generic[Item]):
    """Solution mixin for problems without backlog.

    All backlog quantities are zero.
    """

    def get_backlog_quantity(self, item: Item, period: int) -> int:
        """No backlog."""
        return 0

    def compute_total_backlog_cost(self) -> float:
        """No backlog cost."""
        return 0.0

    def check_backlog_constraints(self) -> bool:
        """Always satisfied (no backlog to check)."""
        return True

    def get_max_backlog(self) -> int:
        """No backlog."""
        return 0

    def get_total_backlog_at_period(self, period: int) -> int:
        """No backlog."""
        return 0
