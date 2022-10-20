#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

from discrete_optimization.coloring.coloring_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.coloring.solvers.coloring_toulbar_solver import (
    ToulbarColoringSolver,
)


def run_toulbar_coloring():
    logging.basicConfig(level=logging.DEBUG)
    file = [f for f in get_data_available() if "gc_70_9" in f][0]
    color_problem = parse_file(file)
    solver = ToulbarColoringSolver(color_problem, params_objective_function=None)
    solver.init_model(
        value_sequence_chain=True,
        hard_value_sequence_chain=False,
        tolerance_delta_max=2,
    )
    result_store = solver.solve(time_limit=100)
    solution = result_store.get_best_solution_fit()[0]
    assert color_problem.satisfy(solution)


if __name__ == "__main__":
    run_toulbar_coloring()
