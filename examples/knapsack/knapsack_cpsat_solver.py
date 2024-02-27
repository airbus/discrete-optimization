#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.knapsack.knapsack_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.knapsack.solvers.knapsack_cpsat_solver import (
    CPSatKnapsackSolver,
)


def run_cpsat_knapsack():
    file = [f for f in get_data_available() if "ks_1000_0" in f][0]
    knapsack_model = parse_file(file)
    cp_model = CPSatKnapsackSolver(knapsack_model)
    cp_model.init_model()
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 10
    result_storage = cp_model.solve(parameters_cp=parameters_cp)
    sol, fit = result_storage.get_best_solution_fit()
    print("Status solver :", cp_model.get_status_solver())
    assert knapsack_model.satisfy(sol)


if __name__ == "__main__":
    run_cpsat_knapsack()
