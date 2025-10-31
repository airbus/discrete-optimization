#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest

from discrete_optimization.binpack.problem import (
    BinPackProblem,
    BinPackSolution,
    ItemBinPack,
)


@pytest.fixture
def problem():
    return BinPackProblem(
        list_items=[ItemBinPack(index=i, weight=i + 1) for i in range(0, 9)],
        capacity_bin=9,
        incompatible_items={(1, 2), (1, 8)},
    )


@pytest.fixture
def manual_sol(problem):
    return BinPackSolution(problem=problem, allocation=[0, 1, 2, 3, 3, 2, 1, 0, 4])


@pytest.fixture
def manual_sol2(problem):
    return BinPackSolution(problem=problem, allocation=[4, 1, 2, 3, 3, 2, 1, 4, 0])
