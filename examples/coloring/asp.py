#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

from discrete_optimization.coloring.parser import get_data_available, parse_file
from discrete_optimization.coloring.plot import plot_coloring_solution, plt
from discrete_optimization.coloring.problem import (
    ColoringConstraints,
    transform_coloring_problem,
)
from discrete_optimization.coloring.solvers.asp import AspColoringSolver


def run_asp_coloring():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "gc_4_1" in f][0]
    color_problem = parse_file(file)
    solver = AspColoringSolver(color_problem, params_objective_function=None)
    solver.init_model(max_models=50, nb_colors=20)
    result_store = solver.solve(time_limit=5)
    solution, fit = result_store.get_best_solution_fit()
    plot_coloring_solution(solution)
    plt.show()
    print(solution, fit)
    print("Evaluation : ", color_problem.evaluate(solution))
    print("Satisfy : ", color_problem.satisfy(solution))


def run_asp_coloring_with_constraints():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "gc_20_1" in f][0]
    color_problem = parse_file(file)
    color_problem = transform_coloring_problem(
        color_problem,
        subset_nodes=set(range(10)),
        constraints_coloring=ColoringConstraints(color_constraint={0: 0, 1: 1, 2: 2}),
    )
    solver = AspColoringSolver(color_problem, params_objective_function=None)
    solver.init_model(max_models=50, nb_colors=20)
    result_store = solver.solve(time_limit=5)
    solution, fit = result_store.get_best_solution_fit()
    plot_coloring_solution(solution)
    plt.show()
    print(solution, fit)
    print("Evaluation : ", color_problem.evaluate(solution))
    print("Satisfy : ", color_problem.satisfy(solution))


if __name__ == "__main__":
    run_asp_coloring_with_constraints()
