import logging

from discrete_optimization.vrp.solver.greedy_vrp import GreedyVRPSolver
from discrete_optimization.vrp.solver.solver_ortools import (
    FirstSolutionStrategy,
    LocalSearchMetaheuristic,
    VrpORToolsSolver,
)
from discrete_optimization.vrp.vrp_parser import (
    get_data_available,
    parse_file,
    parse_input,
)
from discrete_optimization.vrp.vrp_solvers import VRPIterativeLP, VRPIterativeLP_Pymip


def test_ortools_vrp_solver():
    logging.basicConfig(level=logging.ERROR)
    file_path = get_data_available()[0]
    vrp_model = parse_file(file_path)
    solver = VrpORToolsSolver(vrp_model=vrp_model)
    solver.init_model(
        first_solution_strategy=FirstSolutionStrategy.SAVINGS,
        local_search_metaheuristic=LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
    )
    res = solver.solve(time_limit_seconds=20)
    sol, fit = res.get_best_solution_fit()
    assert vrp_model.satisfy(sol)


def test_greedy_vrp_solver():
    logging.basicConfig(level=logging.ERROR)
    print(get_data_available())
    file_path = [f for f in get_data_available() if "vrp_31_9_1" in f][0]
    print(file_path)
    vrp_model = parse_file(file_path)
    greedy_solver = GreedyVRPSolver(vrp_model=vrp_model, params_objective_function=None)
    res = greedy_solver.solve(limit_time_s=20)
    sol, fit = res.get_best_solution_fit()
    assert vrp_model.satisfy(sol)
