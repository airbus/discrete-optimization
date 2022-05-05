import random
from typing import Dict, Hashable, Tuple

import pytest
from discrete_optimization.coloring.coloring_model import (
    ColoringProblem,
    ColoringSolution,
)
from discrete_optimization.coloring.coloring_parser import (
    files_available,
    parse,
    parse_file,
)
from discrete_optimization.coloring.coloring_solvers import (
    ColoringLP,
    look_for_solver,
    look_for_solver_class,
    solve,
    solvers_compatibility,
    solvers_map,
)


@pytest.mark.parametrize("coloring_problem_file", files_available)
def test_load_file(coloring_problem_file):
    coloring_model: ColoringProblem = parse_file(coloring_problem_file)
    dummy_solution = coloring_model.get_dummy_solution()
    assert coloring_model.satisfy(dummy_solution)


def test_solvers():
    print(files_available)
    small_example = [f for f in files_available if "gc_20_1" in f][0]
    coloring_model: ColoringProblem = parse_file(small_example)
    solvers = solvers_map.keys()
    for s in solvers:
        if s == ColoringLP:
            # you need a gurobi licence to test this solver.
            continue
        results = solve(method=s, coloring_model=coloring_model, **solvers_map[s][1])
        s, f = results.get_best_solution_fit()
        print(s, f)


if __name__ == "__main__":
    test_solvers()
