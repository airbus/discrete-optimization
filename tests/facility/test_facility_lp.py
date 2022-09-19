#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os

import pytest

from discrete_optimization.facility.facility_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.facility.solvers.facility_lp_solver import (
    LP_Facility_Solver,
    LP_Facility_Solver_CBC,
    LP_Facility_Solver_PyMip,
    MilpSolverName,
    ParametersMilp,
)
from discrete_optimization.generic_tools.do_problem import get_default_objective_setup

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_facility_lp_gurobi():
    file = [f for f in get_data_available() if os.path.basename(f) == "fl_3_1"][0]
    color_problem = parse_file(file)
    solver = LP_Facility_Solver(color_problem)
    parameters_lp = ParametersMilp.default()
    parameters_lp.time_limit = 20
    solution, fit = solver.solve(
        parameters_milp=parameters_lp,
        use_matrix_indicator_heuristic=False,
    ).get_best_solution_fit()
    assert color_problem.satisfy(solution)


def test_facility_lp_cbc():
    file = [f for f in get_data_available() if os.path.basename(f) == "fl_100_7"][0]
    color_problem = parse_file(file)
    solver = LP_Facility_Solver_CBC(color_problem)
    parameters_lp = ParametersMilp.default()
    parameters_lp.time_limit = 20
    solution, fit = solver.solve(
        parameters_milp=parameters_lp,
        use_matrix_indicator_heuristic=False,
    ).get_best_solution_fit()
    assert color_problem.satisfy(solution)


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_facility_lp_pymip():
    file = [f for f in get_data_available() if os.path.basename(f) == "fl_100_7"][0]
    facility_problem = parse_file(file)
    params_objective_function = get_default_objective_setup(problem=facility_problem)
    parameters_lp = ParametersMilp.default()
    parameters_lp.time_limit = 20
    solver = LP_Facility_Solver_PyMip(
        facility_problem,
        milp_solver_name=MilpSolverName.CBC,
        params_objective_function=params_objective_function,
    )
    result_store = solver.solve(
        parameters_milp=parameters_lp,
        use_matrix_indicator_heuristic=False,
    )
    solution = result_store.get_best_solution_fit()[0]
    assert facility_problem.satisfy(solution)
    facility_problem.evaluate(solution)


if __name__ == "__main__":
    test_facility_lp_cbc()
