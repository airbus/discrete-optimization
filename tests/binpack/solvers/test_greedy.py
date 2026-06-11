#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest

from discrete_optimization.binpack.solvers.greedy import (
    BinSelectionStrategy,
    GreedyBinPackSolver,
    SortingStrategy,
)


def test_greedy_default(problem):
    """Test greedy solver with default hyperparameters."""
    solver = GreedyBinPackSolver(problem=problem)
    res = solver.solve()
    sol = res[-1][0]
    assert problem.satisfy(sol)
    # Default is BFD which should be quite good
    assert len(res) == 1


@pytest.mark.parametrize(
    "sorting_strategy,bin_selection_strategy",
    [
        # Test all sorting strategies with FIRST_FIT
        (SortingStrategy.NONE, BinSelectionStrategy.FIRST_FIT),
        (SortingStrategy.WEIGHT_DESC, BinSelectionStrategy.FIRST_FIT),
        (SortingStrategy.WEIGHT_DESC_CONFLICT_ASC, BinSelectionStrategy.FIRST_FIT),
        (SortingStrategy.CONFLICT_DESC_WEIGHT_DESC, BinSelectionStrategy.FIRST_FIT),
        # Test all sorting strategies with BEST_FIT_MIN_WEIGHT
        (SortingStrategy.NONE, BinSelectionStrategy.BEST_FIT_MIN_WEIGHT),
        (SortingStrategy.WEIGHT_DESC, BinSelectionStrategy.BEST_FIT_MIN_WEIGHT),
        (
            SortingStrategy.WEIGHT_DESC_CONFLICT_ASC,
            BinSelectionStrategy.BEST_FIT_MIN_WEIGHT,
        ),
        (
            SortingStrategy.CONFLICT_DESC_WEIGHT_DESC,
            BinSelectionStrategy.BEST_FIT_MIN_WEIGHT,
        ),
        # Test all sorting strategies with BEST_FIT_MIN_REMAINING
        (SortingStrategy.NONE, BinSelectionStrategy.BEST_FIT_MIN_REMAINING),
        (SortingStrategy.WEIGHT_DESC, BinSelectionStrategy.BEST_FIT_MIN_REMAINING),
        (
            SortingStrategy.WEIGHT_DESC_CONFLICT_ASC,
            BinSelectionStrategy.BEST_FIT_MIN_REMAINING,
        ),
        (
            SortingStrategy.CONFLICT_DESC_WEIGHT_DESC,
            BinSelectionStrategy.BEST_FIT_MIN_REMAINING,
        ),
    ],
)
def test_greedy_strategies(problem, sorting_strategy, bin_selection_strategy):
    """Test greedy solver with different strategy combinations."""
    solver = GreedyBinPackSolver(problem=problem)
    res = solver.solve(
        sorting_strategy=sorting_strategy,
        bin_selection_strategy=bin_selection_strategy,
    )
    sol = res[-1][0]

    # Check solution is valid
    assert problem.satisfy(sol), (
        f"Solution not valid for {sorting_strategy.name} + {bin_selection_strategy.name}"
    )

    # Check we got a solution
    assert len(res) == 1

    # Check all items are assigned
    assert len(sol.allocation) == problem.nb_items
    assert all(alloc is not None for alloc in sol.allocation), (
        f"Some items not assigned for {sorting_strategy.name} + {bin_selection_strategy.name}"
    )


def test_greedy_incompatibilities(problem):
    """Test that incompatibility constraints are respected."""
    solver = GreedyBinPackSolver(problem=problem)
    # Try different strategies
    for sorting in SortingStrategy:
        for bin_sel in BinSelectionStrategy:
            res = solver.solve(sorting_strategy=sorting, bin_selection_strategy=bin_sel)
            sol = res[-1][0]

            # Check incompatibilities are respected
            for item_i, item_j in problem.incompatible_items:
                bin_i = sol.allocation[item_i]
                bin_j = sol.allocation[item_j]
                assert bin_i != bin_j, (
                    f"Items {item_i} and {item_j} are incompatible but both in bin {bin_i} "
                    f"with {sorting.name} + {bin_sel.name}"
                )


def test_greedy_capacity(problem):
    """Test that bin capacity constraints are respected."""
    solver = GreedyBinPackSolver(problem=problem)
    res = solver.solve()
    sol = res[-1][0]

    # Calculate weight per bin
    from collections import defaultdict

    bin_weights = defaultdict(float)
    for item_idx, bin_id in enumerate(sol.allocation):
        bin_weights[bin_id] += problem.list_items[item_idx].weight

    # Check all bins respect capacity
    for bin_id, weight in bin_weights.items():
        assert weight <= problem.capacity_bin, (
            f"Bin {bin_id} exceeds capacity: {weight} > {problem.capacity_bin}"
        )
