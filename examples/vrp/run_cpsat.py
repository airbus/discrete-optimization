#  Copyright (c) 2024-2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from matplotlib import pyplot as plt

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.vrp.parser import get_data_available, parse_file
from discrete_optimization.vrp.plot import plot_vrp_solution
from discrete_optimization.vrp.problem import VrpSolution
from discrete_optimization.vrp.solvers.cpsat import CpSatVrpSolver

logging.basicConfig(level=logging.INFO)
# {'nb_vehicles': 25, 'max_length': 633.2618339375006, 'length': 10237.95993753123, 'capacity_violation': 0.0}


def run_cpsat_vrp():
    file = [f for f in get_data_available() if "vrp_26_8_1" in f][0]
    problem = parse_file(file_path=file, start_index=0, end_index=0)
    problem.vehicle_capacities = [
        problem.vehicle_capacities[i] for i in range(problem.vehicle_count)
    ]
    print(problem)
    solver = CpSatVrpSolver(problem=problem)
    solver.init_model(optional_node=False, cut_transition=False)
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    res = solver.solve(parameters_cp=p, time_limit=20)
    print(solver.status_solver)
    sol, fit = res.get_best_solution_fit()
    sol: VrpSolution
    print(problem.evaluate(sol))
    assert problem.satisfy(sol)
    plot_vrp_solution(vrp_problem=problem, solution=sol)
    plt.show()


def run_cpsat_vrp_on_tsp():
    from discrete_optimization.tsp.parser import (
        Point2DTspProblem,
        get_data_available,
        parse_file,
    )
    from discrete_optimization.tsp.problem import TspSolution
    from discrete_optimization.vrp.problem import Customer2D, Customer2DVrpProblem

    file = [f for f in get_data_available() if "tsp_200_1" in f][0]
    problem_tsp: Point2DTspProblem = parse_file(
        file_path=file, start_index=0, end_index=0
    )
    problem = Customer2DVrpProblem(
        vehicle_count=1,
        vehicle_capacities=[100000],
        customer_count=problem_tsp.node_count,
        customers=[
            Customer2D(
                name=str(i),
                demand=0,
                x=problem_tsp.list_points[i].x,
                y=problem_tsp.list_points[i].y,
            )
            for i in range(len(problem_tsp.list_points))
        ],
        start_indexes=[problem_tsp.start_index],
        end_indexes=[problem_tsp.end_index],
    )
    print(problem)
    solver = CpSatVrpSolver(problem=problem)
    solver.init_model(optional_node=False, cut_transition=False)
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    res = solver.solve(parameters_cp=p, time_limit=10)
    sol, fit = res.get_best_solution_fit()
    sol: VrpSolution
    print(problem.evaluate(sol))
    plot_vrp_solution(vrp_problem=problem, solution=sol)
    sol_tsp = TspSolution(
        problem=problem_tsp,
        start_index=problem_tsp.start_index,
        end_index=problem_tsp.end_index,
        permutation=sol.list_paths[0],
    )
    assert problem_tsp.satisfy(sol_tsp)
    plt.show()


def warm_starting():
    file = [f for f in get_data_available() if "vrp_26_8_1" in f][0]
    problem = parse_file(file_path=file)
    solver = CpSatVrpSolver(problem=problem)
    solver.init_model(optional_node=False, cut_transition=False)
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    res = solver.solve(parameters_cp=p, time_limit=10)
    sol, fit = res.get_best_solution_fit()
    sol: VrpSolution
    print(problem.evaluate(sol))

    # test warm start
    # start_solution = GreedyVrpSolver(problem=vrp_problem).solve(time_limit=20).get_best_solution_fit()[0]
    start_solution = res[1][0]
    print(start_solution.list_paths)
    # warm start at first solution
    solver = CpSatVrpSolver(problem=problem)
    solver.init_model(optional_node=False, cut_transition=False)
    hints = solver.set_warm_start(start_solution)
    print(hints)
    # force first solution to be the hinted one
    res = solver.solve(
        parameters_cp=p,
        ortools_cpsat_solver_kwargs=dict(
            fix_variables_to_their_hinted_value=False, log_search_progress=True
        ),
    )
    print(solver.status_solver)
    # assert res[0][0].list_paths == start_solution.list_paths


if __name__ == "__main__":
    run_cpsat_vrp()
