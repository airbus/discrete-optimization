#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import pytest

from discrete_optimization.binpack.solvers.dp import DpBinPackSolver, ModelingDpBinPack


@pytest.mark.parametrize(
    "modeling",
    [ModelingDpBinPack.ASSIGN_ITEM_BINS, ModelingDpBinPack.PACK_ITEMS],
)
def test_dp(modeling, problem):
    solver = DpBinPackSolver(problem=problem)
    solver.init_model(
        upper_bound=problem.nb_items,
        modeling=modeling,
    )
    res = solver.solve()
    sol = res[-1][0]
    assert problem.satisfy(sol)
