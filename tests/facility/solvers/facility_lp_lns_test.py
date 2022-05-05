import os

from discrete_optimization.facility.facility_parser import files_available, parse_file
from discrete_optimization.facility.solvers.facility_lp_lns_solver import (
    ConstraintHandlerFacility,
    InitialFacilityMethod,
    InitialFacilitySolution,
)
from discrete_optimization.facility.solvers.facility_lp_solver import (
    LP_Facility_Solver,
    LP_Facility_Solver_CBC,
    LP_Facility_Solver_PyMip,
    MilpSolverName,
    ParametersMilp,
)
from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.generic_tools.lns_mip import LNS_MILP


def facility_lns():
    file = [f for f in files_available if os.path.basename(f) == "fl_100_1"][0]
    print(file)
    facility_problem = parse_file(file)
    params_objective_function = get_default_objective_setup(problem=facility_problem)
    params_milp = ParametersMilp(
        time_limit=300,
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
    solver.init_model(use_matrix_indicator_heuristic=False)
    initial_solution_provider = InitialFacilitySolution(
        problem=facility_problem,
        initial_method=InitialFacilityMethod.GREEDY,
        params_objective_function=params_objective_function,
    )
    constraint_handler = ConstraintHandlerFacility(
        problem=facility_problem, fraction_to_fix=0.5, skip_first_iter=True
    )
    lns_solver = LNS_MILP(
        problem=facility_problem,
        milp_solver=solver,
        initial_solution_provider=initial_solution_provider,
        constraint_handler=constraint_handler,
        params_objective_function=params_objective_function,
    )

    result_store = lns_solver.solve_lns(
        parameters_milp=params_milp, nb_iteration_lns=100
    )
    solution = result_store.get_best_solution_fit()[0]
    print(solution)
    print("Satisfy : ", facility_problem.satisfy(solution))
    print(facility_problem.evaluate(solution))


if __name__ == "__main__":
    facility_lns()
