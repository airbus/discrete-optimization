import logging

from discrete_optimization.vrp.parser import get_data_available, parse_file
from discrete_optimization.vrp.solvers.greedy import GreedyVrpSolver
from discrete_optimization.vrp.solvers.ortools_routing import (
    FirstSolutionStrategy,
    LocalSearchMetaheuristic,
    OrtoolsVrpSolver,
)
from discrete_optimization.vrp.solvers_map import LPIterativeVrpSolver


def run_ortools_vrp_solver():
    logging.basicConfig(level=logging.ERROR)
    file_path = get_data_available()[0]
    print(file_path)
    vrp_problem = parse_file(file_path)
    print("Nb vehicle : ", vrp_problem.vehicle_count)
    print("Capacities : ", vrp_problem.vehicle_capacities)
    solver = OrtoolsVrpSolver(problem=vrp_problem)
    solver.init_model(
        first_solution_strategy=FirstSolutionStrategy.SAVINGS,
        local_search_metaheuristic=LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
    )
    res = solver.solve(time_limit_seconds=20)
    sol, fit = res.get_best_solution_fit()
    print(vrp_problem.evaluate(sol))
    print(vrp_problem.satisfy(sol))


def run_lp_vrp_solver():
    logging.basicConfig(level=logging.ERROR)
    print(get_data_available())
    file_path = [f for f in get_data_available() if "vrp_31_9_1" in f][0]
    print(file_path)
    vrp_problem = parse_file(file_path)
    print("Nb vehicle : ", vrp_problem.vehicle_count)
    print("Capacities : ", vrp_problem.vehicle_capacities)
    solver = LPIterativeVrpSolver(problem=vrp_problem)
    solver.init_model()
    res = solver.solve(time_limit=20)
    sol, fit = res.get_best_solution_fit()
    print(vrp_problem.evaluate(sol))
    print(vrp_problem.satisfy(sol))


def run_greedy_vrp_solver():
    logging.basicConfig(level=logging.ERROR)
    print(get_data_available())
    file_path = [f for f in get_data_available() if "vrp_31_9_1" in f][0]
    print(file_path)
    vrp_problem = parse_file(file_path)
    greedy_solver = GreedyVrpSolver(problem=vrp_problem, params_objective_function=None)
    res = greedy_solver.solve(time_limit=20)
    sol, fit = res.get_best_solution_fit()
    print(vrp_problem.evaluate(sol))
    print(vrp_problem.satisfy(sol))


if __name__ == "__main__":
    run_greedy_vrp_solver()
