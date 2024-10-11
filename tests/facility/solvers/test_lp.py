#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os

import pytest

from discrete_optimization.facility.parser import get_data_available, parse_file
from discrete_optimization.facility.solvers.greedy import GreedyFacilitySolver
from discrete_optimization.facility.solvers.lp import (
    CbcFacilitySolver,
    GurobiFacilitySolver,
    MathOptFacilitySolver,
)

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
@pytest.mark.parametrize("use_matrix_indicator_heuristic", [True, False])
def test_facility_lp_gurobi(use_matrix_indicator_heuristic):
    file = [f for f in get_data_available() if os.path.basename(f) == "fl_3_1"][0]
    color_problem = parse_file(file)
    solver = GurobiFacilitySolver(color_problem)
    kwargs = dict(
        time_limit=20,
        use_matrix_indicator_heuristic=use_matrix_indicator_heuristic,
    )
    result_storage = solver.solve(**kwargs)
    solution, fit = result_storage.get_best_solution_fit()
    assert color_problem.satisfy(solution)

    # test warm start
    start_solution = (
        GreedyFacilitySolver(problem=color_problem).solve().get_best_solution()
    )

    # first solution is not start_solution
    assert (
        result_storage[0][0].facility_for_customers
        != start_solution.facility_for_customers
    )

    # warm start => first solution is start_solution
    solver.set_warm_start(start_solution)
    result_storage = solver.solve(**kwargs)
    assert (
        result_storage[0][0].facility_for_customers
        == start_solution.facility_for_customers
    )


@pytest.mark.parametrize("use_matrix_indicator_heuristic", [True, False])
def test_facility_lp_ortools_mathopt(use_matrix_indicator_heuristic):
    file = [f for f in get_data_available() if os.path.basename(f) == "fl_3_1"][0]
    color_problem = parse_file(file)
    solver = MathOptFacilitySolver(color_problem)
    kwargs = dict(
        time_limit=20,
        use_matrix_indicator_heuristic=use_matrix_indicator_heuristic,
    )
    result_storage = solver.solve(**kwargs)
    solution, fit = result_storage.get_best_solution_fit()
    assert color_problem.satisfy(solution)

    # test warm start
    start_solution = (
        GreedyFacilitySolver(problem=color_problem)
        .solve()
        .get_best_solution()
        # FacilitySolution(problem=color_problem, facility_for_customers=[1,1,2,0])
    )

    # first solution is not start_solution
    assert (
        result_storage[0][0].facility_for_customers
        != start_solution.facility_for_customers
    )

    # warm start => first solution is start_solution
    solver = MathOptFacilitySolver(color_problem)
    solver.init_model(**kwargs)
    solver.set_warm_start(start_solution)
    result_storage = solver.solve(mathopt_enable_output=True, **kwargs)
    assert (
        result_storage[0][0].facility_for_customers
        == start_solution.facility_for_customers
    )


def test_facility_lp_cbc():
    file = [f for f in get_data_available() if os.path.basename(f) == "fl_100_7"][0]
    color_problem = parse_file(file)
    solver = CbcFacilitySolver(color_problem)
    solution, fit = solver.solve(
        time_limit=20,
        use_matrix_indicator_heuristic=False,
    ).get_best_solution_fit()
    assert color_problem.satisfy(solution)


if __name__ == "__main__":
    test_facility_lp_cbc()
