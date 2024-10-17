#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

import matplotlib.pyplot as plt

from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.tsp.parser import get_data_available, parse_file
from discrete_optimization.tsp.plot import plot_tsp_solution
from discrete_optimization.tsp.solvers.cpsat import CpSatTspSolver

logging.basicConfig(level=logging.INFO)


def run_cpsat_solver():
    files = get_data_available()
    files = [f for f in files if "tsp_200_1" in f]
    model = parse_file(files[0], start_index=0, end_index=10)
    params_objective_function = get_default_objective_setup(problem=model)
    solver = CpSatTspSolver(model, params_objective_function=params_objective_function)
    solver.init_model()
    p = ParametersCp.default_cpsat()
    res = solver.solve(
        callbacks=[
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
        parameters_cp=p,
    )
    sol, fitness = res.get_best_solution_fit()
    assert model.satisfy(sol)
    fig, ax = plt.subplots(1)
    for sol, fit in res.list_solution_fits:
        ax.clear()
        plot_tsp_solution(tsp_model=model, solution=sol, ax=ax)
        ax.set_title(f"Length ={fit}")
        plt.pause(1.0)
    plt.show()


if __name__ == "__main__":
    run_cpsat_solver()
