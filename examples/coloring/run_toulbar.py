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
from discrete_optimization.coloring.solvers.greedy import (
    GreedyColoringSolver,
    NxGreedyColoringMethod,
)
from discrete_optimization.coloring.solvers.toulbar import ToulbarColoringSolver


def run_toulbar_coloring():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "gc_100_5" in f][0]
    color_problem = parse_file(file)
    solver = ToulbarColoringSolver(color_problem, params_objective_function=None)
    solver.init_model(
        nb_colors=20,
        value_sequence_chain=False,
        hard_value_sequence_chain=True,
        tolerance_delta_max=1,
    )
    # solver.model.Dump("test.wcsp")
    result_store = solver.solve(time_limit=100)
    solution = result_store.get_best_solution_fit()[0]
    plot_coloring_solution(solution)
    plt.show()
    assert color_problem.satisfy(solution)


def run_toulbar_coloring_with_ws():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "gc_100_5" in f][0]
    color_problem = parse_file(file)
    solv = GreedyColoringSolver(problem=color_problem)
    greedy_sol = solv.solve(
        strategy=NxGreedyColoringMethod.best
    ).get_best_solution_fit()[0]
    solver = ToulbarColoringSolver(color_problem, params_objective_function=None)
    solver.init_model(
        nb_colors=20,
        value_sequence_chain=True,
        hard_value_sequence_chain=True,
        tolerance_delta_max=1,
    )
    solver.set_warm_start(greedy_sol)
    result_store = solver.solve(time_limit=20)
    solution = result_store.get_best_solution_fit()[0]
    plot_coloring_solution(solution)
    plt.show()
    assert color_problem.satisfy(solution)


def run_toulbar_with_constraints():
    file = [f for f in get_data_available() if "gc_50_1" in f][0]
    color_problem = parse_file(file)
    color_problem = transform_coloring_problem(
        color_problem,
        subset_nodes=set(range(10)),
        constraints_coloring=ColoringConstraints(color_constraint={0: 3, 1: 2, 2: 4}),
    )
    solver = ToulbarColoringSolver(color_problem, params_objective_function=None)
    solver.init_model(
        nb_colors=20,
        value_sequence_chain=False,
        hard_value_sequence_chain=False,
        tolerance_delta_max=2,
    )
    result_store = solver.solve(time_limit=10)
    solution = result_store.get_best_solution_fit()[0]
    plot_coloring_solution(solution)
    print("Evaluation : ", color_problem.evaluate(solution))
    print("Satisfy : ", color_problem.satisfy(solution))
    assert color_problem.satisfy(solution)
    plt.show()


if __name__ == "__main__":
    run_toulbar_coloring_with_ws()
