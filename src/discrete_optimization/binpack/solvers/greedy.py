#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from enum import Enum
from typing import Any, Optional

import networkx as nx

from discrete_optimization.binpack.problem import BinPackProblem, BinPackSolution
from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_solver import (
    ParamsObjectiveFunction,
    SolverDO,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)


class SortingStrategy(Enum):
    """Strategy for sorting items before bin assignment.

    - NONE: Process items in original order
    - WEIGHT_DESC: Sort by weight descending
    - WEIGHT_DESC_CONFLICT_ASC: Sort by weight descending, then conflict count ascending
    - CONFLICT_DESC_WEIGHT_DESC: Sort by conflict count descending, then weight descending
    """

    NONE = "none"
    WEIGHT_DESC = "weight_desc"
    WEIGHT_DESC_CONFLICT_ASC = "weight_desc_conflict_asc"
    CONFLICT_DESC_WEIGHT_DESC = "conflict_desc_weight_desc"


class BinSelectionStrategy(Enum):
    """Strategy for selecting which bin to assign an item to.

    - FIRST_FIT: Choose the first bin that fits the item
    - BEST_FIT_MIN_WEIGHT: Choose bin with minimum total weight after adding item
    - BEST_FIT_MIN_REMAINING: Choose bin with minimum remaining capacity after adding item
    """

    FIRST_FIT = "first_fit"
    BEST_FIT_MIN_WEIGHT = "best_fit_min_weight"
    BEST_FIT_MIN_REMAINING = "best_fit_min_remaining"


class GreedyBinPackSolver(SolverDO):
    """Greedy bin packing solver with configurable sorting and bin selection strategies.

    Supports multiple greedy heuristics for bin packing problems with optional
    incompatibility constraints. The behavior is controlled by two strategy hyperparameters.

    Hyperparameters:
        sorting_strategy: How to sort items before assignment
        bin_selection_strategy: How to select bins for items

    Example:
        # Best Fit Decreasing (BFD) heuristic
        solver = GreedyBinPackSolver(problem)
        solver.solve(
            sorting_strategy=SortingStrategy.CONFLICT_DESC_WEIGHT_DESC,
            bin_selection_strategy=BinSelectionStrategy.BEST_FIT_MIN_REMAINING
        )
    """

    problem: BinPackProblem

    hyperparameters = [
        EnumHyperparameter(
            name="sorting_strategy",
            enum=SortingStrategy,
            default=SortingStrategy.CONFLICT_DESC_WEIGHT_DESC,
        ),
        EnumHyperparameter(
            name="bin_selection_strategy",
            enum=BinSelectionStrategy,
            default=BinSelectionStrategy.BEST_FIT_MIN_REMAINING,
        ),
    ]

    def __init__(
        self,
        problem: BinPackProblem,
        params_objective_function: ParamsObjectiveFunction | None = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        # Build incompatibility graph
        graph = nx.Graph()
        graph.add_nodes_from(range(self.problem.nb_items))
        if self.problem.has_constraint:
            graph.add_edges_from(list(self.problem.incompatible_items))
        self.graph = graph
        self.neighbors = {n: set(self.graph.neighbors(n)) for n in self.graph.nodes}

    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        """Solve using greedy bin packing with configured strategies."""
        # Complete kwargs with default hyperparameters
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        sorting_strategy = kwargs["sorting_strategy"]
        bin_selection_strategy = kwargs["bin_selection_strategy"]

        callback = CallbackList(callbacks=callbacks)
        callback.on_solve_start(self)

        nb_items = self.problem.nb_items
        list_items = self.problem.list_items

        # Sort items according to strategy
        item_indices = self._sort_items(sorting_strategy)

        # Initialize data structures
        allocation = [None] * nb_items
        bin_weights = []
        bin_items = []

        # Assign items to bins
        for item_idx in item_indices:
            weight = list_items[item_idx].weight
            neighbors = self.neighbors[item_idx]

            # Find best bin according to strategy
            best_bin = self._select_bin(
                weight, neighbors, bin_weights, bin_items, bin_selection_strategy
            )

            if best_bin != -1:
                # Assign to existing bin
                bin_weights[best_bin] += weight
                bin_items[best_bin].add(item_idx)
                allocation[item_idx] = best_bin
            else:
                # Create new bin
                new_bin = len(bin_weights)
                bin_weights.append(weight)
                bin_items.append({item_idx})
                allocation[item_idx] = new_bin

        # Build and return solution
        sol = BinPackSolution(problem=self.problem, allocation=allocation)
        fit = self.aggreg_from_sol(sol)
        res = self.create_result_storage([(sol, fit)])
        callback.on_solve_end(res, self)
        return res

    def _sort_items(self, sorting_strategy: SortingStrategy) -> list[int]:
        """Sort items according to the given sorting strategy."""
        nb_items = self.problem.nb_items
        list_items = self.problem.list_items

        if sorting_strategy == SortingStrategy.NONE:
            return list(range(nb_items))

        elif sorting_strategy == SortingStrategy.WEIGHT_DESC:
            return sorted(
                range(nb_items), key=lambda i: list_items[i].weight, reverse=True
            )

        elif sorting_strategy == SortingStrategy.WEIGHT_DESC_CONFLICT_ASC:
            return sorted(
                range(nb_items),
                key=lambda i: (list_items[i].weight, -len(self.neighbors[i])),
                reverse=True,
            )

        elif sorting_strategy == SortingStrategy.CONFLICT_DESC_WEIGHT_DESC:
            return sorted(
                range(nb_items),
                key=lambda i: (len(self.neighbors[i]), list_items[i].weight),
                reverse=True,
            )

        return list(range(nb_items))

    def _select_bin(
        self,
        weight: float,
        neighbors: set[int],
        bin_weights: list[float],
        bin_items: list[set[int]],
        bin_selection_strategy: BinSelectionStrategy,
    ) -> int:
        """Select a bin for the item according to the given strategy.

        Returns:
            Bin index to use, or -1 if no existing bin fits
        """
        capacity_bin = self.problem.capacity_bin

        if bin_selection_strategy == BinSelectionStrategy.FIRST_FIT:
            # First bin that fits
            for bin_id in range(len(bin_weights)):
                if bin_weights[bin_id] + weight <= capacity_bin:
                    if not any(n in bin_items[bin_id] for n in neighbors):
                        return bin_id
            return -1

        elif bin_selection_strategy == BinSelectionStrategy.BEST_FIT_MIN_WEIGHT:
            # Bin with minimum total weight after adding item
            best_bin = -1
            min_weight = float("inf")
            for bin_id in range(len(bin_weights)):
                if bin_weights[bin_id] + weight <= capacity_bin:
                    if not any(n in bin_items[bin_id] for n in neighbors):
                        new_weight = bin_weights[bin_id] + weight
                        if new_weight < min_weight:
                            min_weight = new_weight
                            best_bin = bin_id
            return best_bin

        elif bin_selection_strategy == BinSelectionStrategy.BEST_FIT_MIN_REMAINING:
            # Bin with minimum remaining capacity after adding item
            best_bin = -1
            min_remaining = float("inf")
            for bin_id in range(len(bin_weights)):
                if bin_weights[bin_id] + weight <= capacity_bin:
                    if not any(n in bin_items[bin_id] for n in neighbors):
                        remaining = capacity_bin - (bin_weights[bin_id] + weight)
                        if remaining < min_remaining:
                            min_remaining = remaining
                            best_bin = bin_id
            return best_bin

        return -1
