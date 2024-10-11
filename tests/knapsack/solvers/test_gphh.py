#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.knapsack.parser import get_data_available, parse_file
from discrete_optimization.knapsack.problem import KnapsackProblem, from_kp_to_multi
from discrete_optimization.knapsack.solvers.gphh import (
    GphhKnapsackSolver,
    ParametersGphh,
)
from discrete_optimization.knapsack.solvers_map import GreedyBestKnapsackSolver


def test_run_one_example():
    files_available = get_data_available()
    one_file = files_available[10]
    knapsack_problem: KnapsackProblem = parse_file(one_file)
    multidimensional_knapsack = from_kp_to_multi(knapsack_problem)
    trainings = [from_kp_to_multi(parse_file(files_available[i])) for i in range(10)]
    params_gphh = ParametersGphh.default()
    params_gphh.pop_size = 40
    params_gphh.crossover_rate = 0.7
    params_gphh.mutation_rate = 0.1
    params_gphh.n_gen = 50
    params_gphh.min_tree_depth = 1
    params_gphh.max_tree_depth = 5
    gphh_solver = GphhKnapsackSolver(
        training_domains=trainings,
        problem=multidimensional_knapsack,
        params_gphh=params_gphh,
    )
    gphh_solver.init_model()
    rs = gphh_solver.solve()
    sol, fit = rs.get_best_solution_fit()
    for k in range(10, len(files_available)):
        kp = parse_file(files_available[k])
        mdkp = from_kp_to_multi(kp)
        rs = gphh_solver.build_result_storage_for_domain(mdkp)
        GreedyBestKnapsackSolver(kp).solve().get_best_solution_fit()
        rs.get_best_solution_fit()
    gphh_solver.plot_solution(show=False)


if __name__ == "__main__":
    test_run_one_example()
