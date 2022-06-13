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
    print(file)
    color_problem = parse_file(file)
    solver = LP_Facility_Solver(color_problem)
    parameters_lp = ParametersMilp.default()
    parameters_lp.TimeLimit = 20
    solution, fit = solver.solve(
        parameters_milp=parameters_lp,
        use_matrix_indicator_heuristic=False,
        verbose=True,
    ).get_best_solution_fit()
    print(solution)
    print("Satisfy : ", color_problem.satisfy(solution))
    print(fit)


def test_facility_lp_cbc():
    file = [f for f in get_data_available() if os.path.basename(f) == "fl_100_7"][0]
    print(file)
    color_problem = parse_file(file)
    solver = LP_Facility_Solver_CBC(color_problem)
    parameters_lp = ParametersMilp.default()
    parameters_lp.TimeLimit = 20
    solution, fit = solver.solve(
        parameters_milp=parameters_lp,
        use_matrix_indicator_heuristic=False,
        verbose=True,
    ).get_best_solution_fit()
    print(solution)
    print("Satisfy : ", color_problem.satisfy(solution))
    print(fit)


def test_facility_lp_pymip():
    file = [f for f in get_data_available() if os.path.basename(f) == "fl_100_7"][0]
    print(file)
    facility_problem = parse_file(file)
    params_objective_function = get_default_objective_setup(problem=facility_problem)
    parameters_lp = ParametersMilp.default()
    parameters_lp.TimeLimit = 20
    solver = LP_Facility_Solver_PyMip(
        facility_problem,
        milp_solver_name=MilpSolverName.CBC,
        params_objective_function=params_objective_function,
    )
    result_store = solver.solve(
        parameters_milp=parameters_lp,
        use_matrix_indicator_heuristic=False,
        verbose=True,
    )
    solution = result_store.get_best_solution_fit()[0]
    print(solution)
    print("Satisfy : ", facility_problem.satisfy(solution))
    print(facility_problem.evaluate(solution))


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_facility_lp_lns_gurobi():
    file = [f for f in get_data_available() if os.path.basename(f) == "fl_3_1"][0]
    color_problem = parse_file(file)
    solver = LP_Facility_Solver(color_problem)
    parameters_lp = ParametersMilp.default()
    parameters_lp.TimeLimit = 20
    solution, fit = solver.solve_lns(
        use_matrix_indicator_heuristic=False,
        fraction_to_fix_first_iter=0,
        fraction_to_fix=0.3,
        nb_iteration=3,
        greedy_start=True,
        parameters_milp=parameters_lp,
        verbose=True,
    )
    print(solution)
    print("Satisfy : ", color_problem.satisfy(solution))


if __name__ == "__main__":
    test_facility_lp_cbc()
