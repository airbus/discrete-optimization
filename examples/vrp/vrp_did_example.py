#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from matplotlib import pyplot as plt

from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.vrp.plots.plot_vrp import plot_vrp_solution
from discrete_optimization.vrp.solver.did_vrp_solver import DidVrpSolver
from discrete_optimization.vrp.vrp_model import VrpSolution
from discrete_optimization.vrp.vrp_parser import get_data_available, parse_file

logging.basicConfig(level=logging.INFO)


def run_did_vrp():
    file = [f for f in get_data_available() if "vrp_26_8_1" in f][0]
    problem = parse_file(file_path=file, start_index=0, end_index=0)
    problem.vehicle_capacities = [
        problem.vehicle_capacities[i] for i in range(problem.vehicle_count)
    ]
    print(problem)
    solver = DidVrpSolver(problem=problem)
    solver.init_model()
    res = solver.solve()
    sol, fit = res.get_best_solution_fit()
    sol: VrpSolution
    print(problem.evaluate(sol))
    assert problem.satisfy(sol)
    plot_vrp_solution(vrp_model=problem, solution=sol)
    plt.show()


def run_optuna():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "vrp_26_8_1" in f][0]
    problem = parse_file(file_path=file, start_index=0, end_index=0)
    from discrete_optimization.generic_tools.optuna.utils import (
        generic_optuna_experiment_monoproblem,
    )

    study = generic_optuna_experiment_monoproblem(
        problem=problem, solvers_to_test=[DidVrpSolver]
    )


if __name__ == "__main__":
    run_optuna()
