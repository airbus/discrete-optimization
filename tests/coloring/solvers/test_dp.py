#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.coloring.parser import get_data_available, parse_file
from discrete_optimization.coloring.problem import ColoringProblem
from discrete_optimization.coloring.solvers.dp import DpColoringSolver, dp


def test_coloring_dp():
    small_example = [f for f in get_data_available() if "gc_20_1" in f][0]
    problem: ColoringProblem = parse_file(small_example)
    solver = DpColoringSolver(problem=problem)
    res = solver.solve(solver=dp.LNBS, time_limit=5)
    sol, fit = res.get_best_solution_fit()
    assert problem.satisfy(sol)
    print(problem.evaluate(sol))
