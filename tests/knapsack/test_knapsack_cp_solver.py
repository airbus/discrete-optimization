#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.knapsack.knapsack_model import KnapsackSolution
from discrete_optimization.knapsack.knapsack_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.knapsack.knapsack_solvers import (
    CPKnapsackMZN,
    CPKnapsackMZN2,
)


def test_cp_knapsack_1():
    file = [f for f in get_data_available() if "ks_30_0" in f][0]
    knapsack_model = parse_file(file)
    cp_model = CPKnapsackMZN(knapsack_model)
    cp_model.init_model()
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 10
    result_storage = cp_model.solve(parameters_cp=parameters_cp)
    sol, fit = result_storage.get_best_solution_fit()
    assert isinstance(result_storage.list_solution_fits[0][0], KnapsackSolution)
    assert len(result_storage.list_solution_fits) == 3


def test_cp_knapsack_2():
    file = [f for f in get_data_available() if "ks_30_0" in f][0]
    knapsack_model = parse_file(file)
    cp_model = CPKnapsackMZN2(knapsack_model)
    cp_model.init_model()
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 10
    result_storage = cp_model.solve(parameters_cp=parameters_cp)
    sol, fit = result_storage.get_best_solution_fit()
    assert isinstance(result_storage.list_solution_fits[0][0], KnapsackSolution)
    assert len(result_storage.list_solution_fits) == 3


if __name__ == "__main__":
    test_cp_knapsack_1()
    test_cp_knapsack_2()
