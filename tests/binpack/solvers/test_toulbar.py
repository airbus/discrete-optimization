#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest

from discrete_optimization.binpack.solvers.toulbar import ToulbarBinPackSolver

try:
    import pytoulbar2
except ImportError:
    toulbar_available = False
else:
    toulbar_available = True


@pytest.mark.skipif(True, reason="You need Toulbar2 to test this solver.")
def test_toulbar(problem, manual_sol, manual_sol2):
    solve_kwargs = dict()

    solver = ToulbarBinPackSolver(problem=problem)
    solver.init_model(upper_bound=problem.nb_items)
    res = solver.solve(**solve_kwargs)
    sol = res[-1][0]
    assert problem.satisfy(sol)
    # assert len(res)>1   # find 2 solutions but only 1 stored

    # check warm start
    if manual_sol.allocation == sol.allocation:
        # ensure using different sol as warm start
        manual_sol = manual_sol2
    assert manual_sol.allocation != sol.allocation
    solver = ToulbarBinPackSolver(problem=problem)
    solver.init_model(upper_bound=problem.nb_items)
    solver.set_warm_start(manual_sol)
    res = solver.solve(**solve_kwargs)
    sol = res[-1][0]
    assert problem.satisfy(sol)
    assert len(res) == 1  # find directly an optimal solution
    assert res[-1][1] == solver.aggreg_from_sol(manual_sol)
    # do not check solution equality as toulbar2 reindex the bins
    # assert manual_sol.allocation == sol.allocation
