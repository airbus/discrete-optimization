#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.knapsack.parser import get_data_available, parse_file
from discrete_optimization.knapsack.problem import KnapsackSolution
from discrete_optimization.knapsack.solvers_map import (
    Cp2KnapsackSolver,
    CpKnapsackSolver,
)


def test_cp_knapsack_1():
    file = [f for f in get_data_available() if "ks_30_0" in f][0]
    knapsack_problem = parse_file(file)
    cp_model = CpKnapsackSolver(knapsack_problem)
    cp_model.init_model()
    result_storage = cp_model.solve(time_limit=10)
    sol, fit = result_storage.get_best_solution_fit()
    assert isinstance(result_storage[0][0], KnapsackSolution)
    assert len(result_storage) == 3


def test_cp_knapsack_2():
    file = [f for f in get_data_available() if "ks_30_0" in f][0]
    knapsack_problem = parse_file(file)
    cp_model = Cp2KnapsackSolver(knapsack_problem)
    cp_model.init_model()
    result_storage = cp_model.solve(time_limit=10)
    sol, fit = result_storage.get_best_solution_fit()
    assert isinstance(result_storage[0][0], KnapsackSolution)
    assert len(result_storage) == 3


if __name__ == "__main__":
    test_cp_knapsack_1()
    test_cp_knapsack_2()
