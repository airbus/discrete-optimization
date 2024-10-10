#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

import matplotlib.pyplot as plt

from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.tsp.plots.plot_tsp import plot_tsp_solution
from discrete_optimization.tsp.solver.did_tsp_solver import DidTspSolver, dp
from discrete_optimization.tsp.tsp_parser import get_data_available, parse_file

logging.basicConfig(level=logging.INFO)


def run_did_solver():
    files = get_data_available()
    files = [f for f in files if "tsp_574_1" in f]
    model = parse_file(files[0], start_index=0, end_index=0)
    params_objective_function = get_default_objective_setup(problem=model)
    solver = DidTspSolver(model, params_objective_function=params_objective_function)
    solver.init_model()
    res = solver.solve(solver=dp.LNBS, time_limit=25)
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
    run_did_solver()
