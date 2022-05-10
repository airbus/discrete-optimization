from discrete_optimization.coloring.coloring_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.coloring.solvers.coloring_cp_solvers import (
    ColoringCP,
    ColoringCPModel,
    NXGreedyColoringMethod,
)
from discrete_optimization.generic_tools.cp_tools import CPSolverName, ParametersCP
from discrete_optimization.generic_tools.do_problem import get_default_objective_setup


def color_cp():
    file = [f for f in get_data_available() if "gc_1000_1" in f][0]
    color_problem = parse_file(file)
    params_objective_function = get_default_objective_setup(problem=color_problem)
    solver = ColoringCP(
        color_problem,
        params_objective_function=params_objective_function,
        cp_solver_name=CPSolverName.CPOPT,
    )
    solver.init_model(
        greedy_start=True,
        greedy_method=NXGreedyColoringMethod.largest_first,
        object_output=True,
        include_seq_chain_constraint=True,
        cp_model=ColoringCPModel.DEFAULT,
    )
    # solver.export_dzn()
    parameters_cp = ParametersCP(
        time_limit=200,
        pool_solutions=1000,
        free_search=True,
        intermediate_solution=True,
        multiprocess=True,
        nb_process=4,
        all_solutions=False,
        nr_solutions=10000,
    )
    result_store = solver.solve(parameters_cp=parameters_cp, verbose=True)
    solution = result_store.get_best_solution_fit()[0]
    print(solution)
    print("Satisfy : ", color_problem.satisfy(solution))


def color_cp_lns():
    file = [f for f in get_data_available() if "gc_1000_9" in f][0]
    color_problem = parse_file(file)
    params_objective_function = get_default_objective_setup(problem=color_problem)
    solver = ColoringCP(
        color_problem,
        params_objective_function=params_objective_function,
        cp_solver_name=CPSolverName.CPOPT,
    )
    solver.init_model(
        greedy_start=False,
        nb_colors=50,
        object_output=True,
        include_seq_chain_constraint=False,
        cp_model=ColoringCPModel.DEFAULT,
        max_cliques=200,
    )
    solution, fit = solver.solve_lns(
        fraction_to_fix=0.8,
        nb_iteration=1000,
        limit_time_s=100,
        greedy_start=True,
        greedy_method=NXGreedyColoringMethod.random_sequential,
        cp_model=ColoringCPModel.LNS,
        object_output=True,
        verbose=True,
    )
    print(solution)
    print("Satisfy : ", color_problem.satisfy(solution))


if __name__ == "__main__":
    color_cp()
