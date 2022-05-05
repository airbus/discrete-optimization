from discrete_optimization.coloring.coloring_parser import files_available, parse_file
from discrete_optimization.coloring.solvers.coloring_lp_lns_solvers import (
    ConstraintHandlerFixColorsPyMip,
    InitialColoring,
    InitialColoringMethod,
)
from discrete_optimization.coloring.solvers.coloring_lp_solvers import (
    ColoringLP,
    ColoringLP_MIP,
    MilpSolverName,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    get_default_objective_setup,
)
from discrete_optimization.generic_tools.lns_mip import LNS_MILP
from discrete_optimization.generic_tools.lp_tools import ParametersMilp


def run_lns():
    file = [f for f in files_available if "gc_70_7" in f][0]
    color_problem = parse_file(file)
    params_objective_function = get_default_objective_setup(color_problem)
    solver = ColoringLP_MIP(
        color_problem,
        milp_solver_name=MilpSolverName.CBC,
        params_objective_function=params_objective_function,
    )
    solver.init_model(greedy_start=False)
    params_milp = ParametersMilp(
        time_limit=100,
        pool_solutions=1000,
        mip_gap_abs=0.0001,
        mip_gap=0.001,
        retrieve_all_solution=True,
        n_solutions_max=100,
    )
    constraint_handler = ConstraintHandlerFixColorsPyMip(
        problem=color_problem, fraction_to_fix=0.9
    )
    initial_provider = InitialColoring(
        problem=color_problem,
        initial_method=InitialColoringMethod.DUMMY,
        params_objective_function=params_objective_function,
    )
    lns_mip = LNS_MILP(
        problem=color_problem,
        milp_solver=solver,
        initial_solution_provider=initial_provider,
        constraint_handler=constraint_handler,
        params_objective_function=params_objective_function,
    )
    result_store = lns_mip.solve_lns(parameters_milp=params_milp, nb_iteration_lns=100)
    solution = result_store.get_best_solution_fit()[0]
    print(color_problem.satisfy(solution))
    print(color_problem.evaluate(solution))


if __name__ == "__main__":
    run_lns()
