#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import pytest

from discrete_optimization.coloring.parser import get_data_available, parse_file
from discrete_optimization.coloring.problem import ColoringProblem
from discrete_optimization.coloring.solvers.greedy import (
    GreedyColoringSolver,
    NxGreedyColoringMethod,
)
from discrete_optimization.coloring.solvers.toulbar import ToulbarColoringSolver

try:
    import pytoulbar2
except ImportError:
    toulbar_available = False
else:
    toulbar_available = True


@pytest.mark.skipif(True, reason="You need Toulbar2 to test this solver.")
def test_coloring_toulbar():
    small_example = [f for f in get_data_available() if "gc_20_1" in f][0]
    problem: ColoringProblem = parse_file(small_example)
    solver = ToulbarColoringSolver(problem=problem)
    res = solver.solve(time_limit=5)
    sol, fit = res.get_best_solution_fit()
    assert problem.satisfy(sol)
    print(problem.evaluate(sol))


@pytest.mark.skipif(True, reason="You need Toulbar2 to test this solver.")
def test_toulbar_coloring_ws():
    file = [f for f in get_data_available() if "gc_20_7" in f][0]
    color_problem = parse_file(file)
    greedy = GreedyColoringSolver(color_problem)
    sol, _ = greedy.solve(strategy=NxGreedyColoringMethod.best).get_best_solution_fit()
    solver = ToulbarColoringSolver(color_problem)
    solver.init_model(nb_colors=30)
    solver.set_warm_start(sol)
    result_store = solver.solve(
        time_limit=10,
    )
    solution, fit = result_store.get_best_solution_fit()
    assert color_problem.satisfy(solution)
