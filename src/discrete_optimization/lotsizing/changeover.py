#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Sequence-dependent changeover costs mixin for lot sizing problems.

This module provides mixins for problems where the cost of setup depends on
the sequence of production (which item was produced previously).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Generic

from discrete_optimization.lotsizing.base import Item
from discrete_optimization.lotsizing.demands import DemandsProblem, DemandsSolution


class ChangeoverCostsProblem(DemandsProblem[Item], Generic[Item]):
    """Mixin for sequence-dependent changeover costs.

    Relevant for multi-item problems where the order of production matters.
    Changeover cost c_ij is the cost to switch from producing item i to item j.

    This is different from setup costs which are item-specific and time-dependent.
    Changeover costs depend on the production sequence.
    """

    @abstractmethod
    def get_changeover_cost(self, from_item: Item, to_item: Item) -> float:
        """Get sequence-dependent changeover cost.

        Cost incurred when switching production from item i to item j.

        Args:
            from_item: Item produced previously
            to_item: Item to be produced next

        Returns:
            Changeover cost c_ij (non-negative)
        """
        ...

    def has_changeover_costs(self) -> bool:
        """Check if changeover costs are significant.

        Returns:
            True if any changeover cost is non-zero
        """
        # Default: check if any changeover cost is non-zero
        # Can be overridden for efficiency
        for item_i in self.items_list:
            for item_j in self.items_list:
                if self.get_changeover_cost(item_i, item_j) > 0:
                    return True
        return False


class ChangeoverCostsSolution(DemandsSolution[Item], Generic[Item]):
    """Solution mixin for changeover cost computation."""

    problem: ChangeoverCostsProblem[Item]

    @abstractmethod
    def get_production_sequence(self) -> list[tuple[int, Item]]:
        """Get production sequence as list of (period, item) tuples.

        The sequence should be sorted by period and include only periods
        where production actually occurs (setup happens).

        Returns:
            List of (period, item) tuples representing production sequence
        """
        ...

    def compute_total_changeover_cost(self) -> float:
        """Compute total changeover cost based on production sequence.

        Sum of changeover costs for consecutive items in the production sequence.

        Returns:
            Total changeover cost
        """
        sequence = self.get_production_sequence()

        if len(sequence) <= 1:
            return 0.0  # No changeovers

        total_cost = 0.0
        for i in range(len(sequence) - 1):
            _, item_from = sequence[i]
            _, item_to = sequence[i + 1]
            cost = self.problem.get_changeover_cost(item_from, item_to)
            total_cost += cost

        return total_cost

    def get_changeover_count(self) -> int:
        """Get number of changeovers (switches between items).

        Returns:
            Number of times production switches from one item to another
        """
        sequence = self.get_production_sequence()
        if len(sequence) <= 1:
            return 0

        count = 0
        for i in range(len(sequence) - 1):
            _, item1 = sequence[i]
            _, item2 = sequence[i + 1]
            if item1 != item2:
                count += 1

        return count


class WithoutChangeoverCostsProblem(ChangeoverCostsProblem[Item], Generic[Item]):
    """Utility mixin for problems without changeover costs.

    All changeover costs are zero - sequence doesn't matter.
    """

    def get_changeover_cost(self, from_item: Item, to_item: Item) -> float:
        """No changeover cost."""
        return 0.0

    def has_changeover_costs(self) -> bool:
        """No changeover costs."""
        return False


class WithoutChangeoverCostsSolution(ChangeoverCostsSolution[Item], Generic[Item]):
    """Solution mixin for problems without changeover costs."""

    def compute_total_changeover_cost(self) -> float:
        """No changeover cost."""
        return 0.0

    def get_changeover_count(self) -> int:
        """Changeovers don't matter without costs."""
        return 0
