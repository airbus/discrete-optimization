#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Base classes for lot sizing problems.

This module provides minimal base classes following the generic_tasks_tools pattern.
Each problem variant is composed of mixins that add specific features.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Hashable
from typing import Generic, Optional, TypeVar

from discrete_optimization.generic_tools.do_problem import Problem, Solution

# Generic type for items/products
# Usually int, but could be string, enum, or any hashable type
Item = TypeVar("Item", bound=Hashable)


class LotSizingProblem(Problem, Generic[Item]):
    """Minimal base class for all lot sizing problems.

    This class only defines the essential structure common to ALL lot sizing variants:
    - Time horizon (number of periods)
    - Items/products to produce

    All other features (demands, costs, capacity, etc.) are added via mixins.

    Similar to TasksProblem in generic_tasks_tools.
    """

    _map_item_to_index: Optional[dict[Item, int]] = None

    @property
    @abstractmethod
    def horizon(self) -> int:
        """Number of time periods T.

        Periods are indexed from 0 to horizon-1.
        """
        ...

    @property
    @abstractmethod
    def items_list(self) -> list[Item]:
        """List of all items (product types) to schedule production for.

        Returns:
            List of unique item identifiers
        """
        ...

    @property
    def nb_items(self) -> int:
        """Number of different items/products."""
        return len(self.items_list)

    def get_index_from_item(self, item: Item) -> int:
        """Get index of item in items_list.

        This is cached for efficiency when items_list doesn't change.

        Args:
            item: Item identifier

        Returns:
            Index in items_list (0 to nb_items-1)
        """
        if self._map_item_to_index is None:
            self._map_item_to_index = {
                item: i for i, item in enumerate(self.items_list)
            }
        return self._map_item_to_index[item]

    def get_item_from_index(self, i: int) -> Item:
        """Get item from index.

        Args:
            i: Index in items_list

        Returns:
            Item identifier
        """
        return self.items_list[i]

    def update_items_list(self) -> None:
        """Call when items_list is updated to reset cache."""
        self._map_item_to_index = None


class LotSizingSolution(Solution, Generic[Item]):
    """Minimal base class for lot sizing solutions.

    This is the base for all solution types. Specific solution representations
    are added by mixins and concrete implementations.

    Similar to TasksSolution in generic_tasks_tools.
    """

    problem: LotSizingProblem[Item]
