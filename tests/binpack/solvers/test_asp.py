#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.binpack.solvers.asp import (
    AspBinPackingSolver,
)


def test_asp(problem):
    solver = AspBinPackingSolver(problem=problem)
    solver.init_model(upper_bound=20)
    solve_kwargs = dict(
        time_limit=3,
    )
    res = solver.solve(**solve_kwargs)
    sol = res[-1][0]
    assert problem.satisfy(sol)


def test_asp_ws(problem, manual_sol):
    solver = AspBinPackingSolver(problem=problem)
    solver.init_model(upper_bound=20)
    solver.set_warm_start(manual_sol)
    solve_kwargs = dict(
        time_limit=3,
    )
    res = solver.solve(**solve_kwargs)
    sol = res[-1][0]
    assert problem.satisfy(sol)
    assert res[0][0].allocation == manual_sol.allocation
