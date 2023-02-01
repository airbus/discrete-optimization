#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import os

os.environ["DO_SKIP_MZN_CHECK"] = "1"

from discrete_optimization.coloring.coloring_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.coloring.solvers.coloring_asp_solver import ColoringASPSolver


def run_asp_coloring():
    logging.basicConfig(level=logging.DEBUG)
    file = [f for f in get_data_available() if "gc_250_5" in f][0]
    color_problem = parse_file(file)
    solver = ColoringASPSolver(color_problem, params_objective_function=None)
    solver.init_model(max_models=100, nb_colors=None)
    result_store = solver.solve(timeout_seconds=100, use_external_timeout=True)
    solution, fit = result_store.get_best_solution_fit()
    print(solution, fit)
    print("Evaluation : ", color_problem.evaluate(solution))
    print("Satisfy : ", color_problem.satisfy(solution))


if __name__ == "__main__":
    run_asp_coloring()
