import mip
from discrete_optimization.coloring.coloring_parser import files_available, parse_file
from discrete_optimization.coloring.solvers.coloring_lp_solvers import (
    ColoringLP,
    ColoringLP_MIP,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    get_default_objective_setup,
)
from discrete_optimization.generic_tools.lp_tools import ParametersMilp


def test_color_lp_gurobi():
    file = [f for f in files_available if "gc_70_1" in f][0]
    color_problem = parse_file(file)
    solver = ColoringLP(
        color_problem,
        params_objective_function=get_default_objective_setup(color_problem),
    )
    result_store = solver.solve(parameters_milp=ParametersMilp.default(), verbose=True)
    solution = result_store.get_best_solution_fit()[0]
    print(solution)
    print("Satisfy : ", color_problem.satisfy(solution))


def test_color_lp_pymip():
    file = [f for f in files_available if "gc_50_7" in f][0]
    color_problem = parse_file(file)
    solver = ColoringLP_MIP(
        color_problem,
        params_objective_function=get_default_objective_setup(color_problem),
    )
    solver.init_model(greedy_start=True)
    params_milp = ParametersMilp(
        time_limit=100,
        pool_solutions=1000,
        mip_gap_abs=0.0001,
        mip_gap=0.001,
        retrieve_all_solution=True,
        n_solutions_max=100,
    )
    result_store = solver.solve(parameters_milp=params_milp, verbose=True)
    solution = result_store.get_best_solution_fit()[0]
    print("Satisfy : ", color_problem.satisfy(solution))


if __name__ == "__main__":
    test_color_lp_pymip()
