#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

import matplotlib.pyplot as plt

from discrete_optimization.coloring.coloring_model import ColoringProblem
from discrete_optimization.coloring.coloring_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.coloring.coloring_plot import plot_coloring_solution
from discrete_optimization.coloring.coloring_solvers import (
    ColoringLP,
    solve,
    solvers_map,
)


def run_solvers():
    logging.basicConfig(level=logging.INFO)
    small_example = [f for f in get_data_available() if "gc_20_1" in f][0]
    coloring_model: ColoringProblem = parse_file(small_example)
    assert coloring_model.graph is not None
    assert coloring_model.number_of_nodes is not None
    assert coloring_model.graph.nodes_name is not None
    solvers = solvers_map.keys()
    for s in solvers:
        if s == ColoringLP:
            # you need a gurobi licence to test this solver.
            continue
        results = solve(method=s, problem=coloring_model, **solvers_map[s][1])
        sol, fit = results.get_best_solution_fit()
        plot_coloring_solution(sol, name_figure=f"{s.__name__} nb_colors={fit}")
        logging.info(f"solver = {s.__name__} sol={sol}, nb_colors={fit}")
    plt.show()


if __name__ == "__main__":
    run_solvers()
