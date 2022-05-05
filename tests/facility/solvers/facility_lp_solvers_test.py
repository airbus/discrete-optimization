import os

from discrete_optimization.facility.facility_parser import files_available, parse_file
from discrete_optimization.facility.solvers.facility_lp_solver import (
    LP_Facility_Solver,
    LP_Facility_Solver_CBC,
    LP_Facility_Solver_PyMip,
    MilpSolverName,
    ParametersMilp,
)
from discrete_optimization.generic_tools.do_problem import get_default_objective_setup


def test_facility_lp():
    file = [f for f in files_available if "fl_100_1" in f][0]
    file = [f for f in files_available if os.path.basename(f) == "fl_100_1"][0]
    print(file)
    color_problem = parse_file(file)
    solver = LP_Facility_Solver(color_problem)
    solution, fit = solver.solve(
        limit_time_s=100, use_matrix_indicator_heuristic=False, verbose=True
    )
    print(solution)
    print("Satisfy : ", color_problem.satisfy(solution))
    print(fit)


def test_facility_lp_cbc():
    file = [f for f in files_available if os.path.basename(f) == "fl_100_7"][0]
    print(file)
    color_problem = parse_file(file)
    solver = LP_Facility_Solver_CBC(color_problem)
    solution, fit = solver.solve(
        limit_time_s=100, use_matrix_indicator_heuristic=False, verbose=True
    )
    print(solution)
    print("Satisfy : ", color_problem.satisfy(solution))
    print(fit)


def test_facility_lp_pymip():
    file = [f for f in files_available if os.path.basename(f) == "fl_100_7"][0]
    print(file)
    facility_problem = parse_file(file)
    params_objective_function = get_default_objective_setup(problem=facility_problem)
    params_milp = ParametersMilp(
        time_limit=100,
        pool_solutions=1000,
        mip_gap=0.0001,
        mip_gap_abs=0.001,
        retrieve_all_solution=True,
        n_solutions_max=1000,
    )
    solver = LP_Facility_Solver_PyMip(
        facility_problem,
        milp_solver_name=MilpSolverName.CBC,
        params_objective_function=params_objective_function,
    )
    result_store = solver.solve(
        parameters_milp=params_milp, use_matrix_indicator_heuristic=False, verbose=True
    )
    solution = result_store.get_best_solution_fit()[0]
    print(solution)
    print("Satisfy : ", facility_problem.satisfy(solution))
    print(facility_problem.evaluate(solution))


def test_facility_lp_lns():
    file = [f for f in files_available if os.path.basename(f) == "fl_100_1"][0]
    color_problem = parse_file(file)
    solver = LP_Facility_Solver(color_problem)
    solution, fit = solver.solve_lns(
        use_matrix_indicator_heuristic=False,
        fraction_to_fix_first_iter=0,
        fraction_to_fix=0.3,
        nb_iteration=50,
        greedy_start=True,
        limit_time_s=100,
        verbose=True,
    )
    print(solution)
    print("Satisfy : ", color_problem.satisfy(solution))


def test_facility_lp_lns_CBC():
    file = [f for f in files_available if os.path.basename(f) == "fl_100_1"][0]
    color_problem = parse_file(file)
    solver = LP_Facility_Solver_CBC(color_problem)
    solution, fit = solver.solve_lns(
        use_matrix_indicator_heuristic=False,
        fraction_to_fix_first_iter=0,
        fraction_to_fix=0.3,
        nb_iteration=50,
        greedy_start=True,
        limit_time_s=100,
        verbose=True,
    )
    print(solution)
    print("Satisfy : ", color_problem.satisfy(solution))


if __name__ == "__main__":
    test_facility_lp()