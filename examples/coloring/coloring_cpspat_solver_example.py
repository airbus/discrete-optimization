#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import os

from discrete_optimization.coloring.coloring_model import (
    ConstraintsColoring,
    transform_coloring_problem,
)

os.environ["DO_SKIP_MZN_CHECK"] = "1"

import logging

from discrete_optimization.coloring.coloring_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.coloring.coloring_plot import plot_coloring_solution, plt
from discrete_optimization.coloring.solvers.coloring_cpsat_solver import (
    ColoringCPSatSolver,
    ModelingCPSat,
)
from discrete_optimization.generic_tools.callbacks.loggers import NbIterationTracker
from discrete_optimization.generic_tools.cp_tools import ParametersCP

logging.basicConfig(level=logging.INFO)


def run_cpsat_coloring():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "gc_250_3" in f][0]
    color_problem = parse_file(file)
    solver = ColoringCPSatSolver(color_problem, params_objective_function=None)
    solver.init_model(
        modeling=ModelingCPSat.INTEGER,
        warmstart=False,
        value_sequence_chain=False,
        used_variable=True,
        symmetry_on_used=True,
    )
    p = ParametersCP.default_cpsat()
    p.time_limit = 100
    logging.info("Starting solve")
    result_store = solver.solve(
        callbacks=[NbIterationTracker(step_verbosity_level=logging.INFO)],
        parameters_cp=p,
    )
    print("Status solver : ", solver.get_status_solver())
    solution, fit = result_store.get_best_solution_fit()
    plot_coloring_solution(solution)
    plt.show()
    print(solution, fit)
    print("Evaluation : ", color_problem.evaluate(solution))
    print("Satisfy : ", color_problem.satisfy(solution))


def run_cpsat_coloring_with_constraints():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "gc_20_1" in f][0]
    color_problem = parse_file(file)
    color_problem = transform_coloring_problem(
        color_problem,
        subset_nodes=set(range(10)),
        constraints_coloring=ConstraintsColoring(color_constraint={0: 0, 1: 1, 2: 2}),
    )
    solver = ColoringCPSatSolver(color_problem)
    solver.init_model(nb_colors=20)
    p = ParametersCP.default()
    p.time_limit = 20
    result_store = solver.solve(parameters_cp=p)
    solution, fit = result_store.get_best_solution_fit()
    print("Status solver : ", solver.get_status_solver())
    plot_coloring_solution(solution)
    plt.show()
    print(solution, fit)
    print("Evaluation : ", color_problem.evaluate(solution))
    print("Satisfy : ", color_problem.satisfy(solution))


if __name__ == "__main__":
    run_cpsat_coloring()
