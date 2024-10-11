#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from matplotlib import pyplot as plt

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
    print(problem)
    solver = DpVrpSolver(problem=problem)
    solver.init_model()
    res = solver.solve(solver=dp.CABS, threads=6, time_limit=20)
    sol, fit = res.get_best_solution_fit()
    sol: VrpSolution
    print(problem.evaluate(sol))
    assert problem.satisfy(sol)
    print(sol.list_paths)
    plot_vrp_solution(vrp_problem=problem, solution=sol)
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
