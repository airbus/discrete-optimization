#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.coloring.coloring_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.coloring.coloring_plot import plot_coloring_solution, plt
from discrete_optimization.coloring.solvers.coloring_lp_solvers import ColoringLP
from discrete_optimization.generic_tools.callbacks.loggers import NbIterationTracker
from discrete_optimization.generic_tools.lp_tools import ParametersMilp


def run_gurobi_coloring():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "gc_250_1" in f][0]
    color_problem = parse_file(file)
    solver = ColoringLP(color_problem)
    solver.init_model()
    p = ParametersMilp.default()
    result_store = solver.solve(
        callbacks=[
            NbIterationTracker(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
        parameters_milp=p,
    )
    solution, fit = result_store.get_best_solution_fit()
    plot_coloring_solution(solution)
    plt.show()
    print(solution, fit)
    print("Evaluation : ", color_problem.evaluate(solution))
    print("Satisfy : ", color_problem.satisfy(solution))


if __name__ == "__main__":
    run_gurobi_coloring()
