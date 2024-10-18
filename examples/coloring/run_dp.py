#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.coloring.parser import get_data_available, parse_file
from discrete_optimization.coloring.problem import (
    transform_color_values_to_value_precede_on_other_node_order,
)
from discrete_optimization.coloring.solvers.dp import (
    DpColoringModeling,
    DpColoringSolver,
    dp,
)
from discrete_optimization.coloring.solvers.greedy import (
    GreedyColoringSolver,
    NxGreedyColoringMethod,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
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


def run_dp_coloring_ws():
    logging.basicConfig(level=logging.INFO)
    do_ws = True
    file = [f for f in get_data_available() if "gc_500_7" in f][0]
    color_problem = parse_file(file)
    greedy = GreedyColoringSolver(color_problem)
    sol, _ = greedy.solve(
        strategy=NxGreedyColoringMethod.dsatur
    ).get_best_solution_fit()
    solver = DpColoringSolver(color_problem)
    solver.init_model(modeling=DpColoringModeling.COLOR_TRANSITION, nb_colors=120)
    if do_ws:
        solver.set_warm_start(sol)
    result_store = solver.solve(
        solver=dp.LNBS,
        callbacks=[NbIterationStopper(nb_iteration_max=20)],
        threads=5,
        retrieve_intermediate_solutions=True,
        time_limit=100,
    )
    solution, fit = result_store.get_best_solution_fit()
    trans_ws = transform_color_values_to_value_precede_on_other_node_order(
        sol.colors, nodes_ordering=solver.nodes_reordering
    )
    if do_ws:
        print(trans_ws, solution.colors)
        assert result_store[0][0].colors == trans_ws
    print(solution, fit)
    print("Evaluation : ", color_problem.evaluate(solution))
    print("Satisfy : ", color_problem.satisfy(solution))
    # plot_coloring_solution(solution)
    # plt.show()


if __name__ == "__main__":
    run_dp_coloring()
    run_dp_coloring_ws()
