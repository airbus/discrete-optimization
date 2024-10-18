#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from matplotlib import pyplot as plt

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.vrp.parser import get_data_available, parse_file
from discrete_optimization.vrp.plot import plot_vrp_solution
from discrete_optimization.vrp.problem import VrpSolution
from discrete_optimization.vrp.solvers.dp import DpVrpSolver, dp

logging.basicConfig(level=logging.INFO)


def run_dp_vrp():
    file = [f for f in get_data_available() if "vrp_135_7_1" in f][0]
    problem = parse_file(file_path=file, start_index=0, end_index=0)
    problem.vehicle_capacities = [
        problem.vehicle_capacities[i] for i in range(problem.vehicle_count)
    ]
    solver = DpVrpSolver(problem=problem)
    solver.init_model()
    res = solver.solve(solver=dp.CABS, threads=6, time_limit=30)
    sol, fit = res.get_best_solution_fit()
    sol: VrpSolution
    print(problem.evaluate(sol))
    assert problem.satisfy(sol)
    print(sol.list_paths)
    fig, ax = plt.subplots(1)
    for sol, fit in res.list_solution_fits:
        ax.clear()
        plot_vrp_solution(vrp_problem=problem, solution=sol, ax=ax)
        ax.set_title(f"Length ={fit}")
        plt.pause(0.2)
    plt.show()


def run_dp_vrp_ws():
    from discrete_optimization.vrp.solvers.ortools_routing import OrtoolsVrpSolver

    file = [f for f in get_data_available() if "vrp_200_16_1" in f][0]
    problem = parse_file(file_path=file, start_index=0, end_index=0)
    problem.vehicle_capacities = [
        problem.vehicle_capacities[i] for i in range(problem.vehicle_count)
    ]
    solver = OrtoolsVrpSolver(problem)
    sol_ws = solver.solve(time_limit=5, verbose=True)[0][0]
    solver = DpVrpSolver(problem=problem)
    solver.init_model()
    solver.set_warm_start(sol_ws)
    res = solver.solve(
        solver=dp.LNBS,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        retrieve_intermediate_solutions=True,
        threads=6,
        time_limit=20,
    )
    sol, fit = res.get_best_solution_fit()
    sol: VrpSolution
    assert sol_ws.list_paths == res[0][0].list_paths
    print(problem.evaluate(sol))
    assert problem.satisfy(sol)
    print(sol.list_paths)
    fig, ax = plt.subplots(1)
    for sol, fit in res.list_solution_fits:
        ax.clear()
        plot_vrp_solution(vrp_model=problem, solution=sol, ax=ax)
        ax.set_title(f"Length ={fit}")
        plt.pause(0.2)
    plt.show()


def run_optuna():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "vrp_26_8_1" in f][0]
    problem = parse_file(file_path=file, start_index=0, end_index=0)
    from discrete_optimization.generic_tools.optuna.utils import (
        generic_optuna_experiment_monoproblem,
    )

    study = generic_optuna_experiment_monoproblem(
        problem=problem, solvers_to_test=[DpVrpSolver]
    )


if __name__ == "__main__":
    run_dp_vrp()
