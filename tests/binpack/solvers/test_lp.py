#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.binpack.solvers.lp import MathOptBinPackSolver


def test_lp(problem, manual_sol, manual_sol2):
    solver = MathOptBinPackSolver(problem=problem)
    solver.init_model(upper_bound=problem.nb_items)
    solve_kwargs = dict()
    res = solver.solve(**solve_kwargs)
    sol = res[-1][0]
    assert problem.satisfy(sol)

    # check warm start
    if manual_sol.allocation == sol.allocation:
        # ensure using different sol as warm start
        manual_sol = manual_sol2
    assert manual_sol.allocation != sol.allocation
    solver.set_warm_start(manual_sol)
    res = solver.solve(**solve_kwargs)
    sol = res[-1][0]
    assert problem.satisfy(sol)
    assert manual_sol.allocation == sol.allocation
