#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.coloring.parser import get_data_available, parse_file
from discrete_optimization.coloring.plot import plot_coloring_solution, plt
from discrete_optimization.coloring.solvers.dp import (
    DpColoringModeling,
    DpColoringSolver,
    dp,
)


def run_dp_coloring():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "gc_500_7" in f][0]
    color_problem = parse_file(file)
    solver = DpColoringSolver(color_problem)
    solver.init_model(modeling=DpColoringModeling.COLOR_TRANSITION, nb_colors=120)
    result_store = solver.solve(solver=dp.CABS, threads=10, time_limit=100)
    solution, fit = result_store.get_best_solution_fit()
    print(solution, fit)
    print("Evaluation : ", color_problem.evaluate(solution))
    print("Satisfy : ", color_problem.satisfy(solution))
    # plot_coloring_solution(solution)
    # plt.show()


if __name__ == "__main__":
    run_dp_coloring()