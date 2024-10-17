#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

from matplotlib import pyplot as plt

from discrete_optimization.tsp.parser import get_data_available, parse_file
from discrete_optimization.tsp.plot import plot_tsp_solution
from discrete_optimization.tsp.solvers.gpdp import GpdpBasedTspSolver

logging.basicConfig(level=logging.INFO)


def run_gpdp():
    files = get_data_available()
    files = [f for f in files if "tsp_724_1" in f]
    model = parse_file(files[0], start_index=0, end_index=0)
    solver = GpdpBasedTspSolver(problem=model)
    res = solver.solve(time_limit=10)
    sol, fit = res.get_best_solution_fit()
    assert model.satisfy(sol)
    fig, ax = plt.subplots(1)
    list_solution_fit = sorted(res.list_solution_fits, key=lambda x: x[1], reverse=True)
    for sol, fit in list_solution_fit:
        ax.clear()
        plot_tsp_solution(tsp_model=model, solution=sol, ax=ax)
        ax.set_title(f"Length ={fit}")
        plt.pause(0.05)
    plt.show()


if __name__ == "__main__":
    run_gpdp()
