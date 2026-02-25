#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.knapsack.parser import get_data_available, parse_file
from discrete_optimization.knapsack.solvers.optal import OptalKnapsackSolver


def run_cpsat_knapsack():
    file = [f for f in get_data_available() if "ks_1000_0" in f][0]
    knapsack_problem = parse_file(file)
    cp_model = OptalKnapsackSolver(knapsack_problem)
    cp_model.init_model()
    params_cp = ParametersCp.default()
    result_storage = cp_model.solve(parameters_cp=params_cp, time_limit=10)
    sol, fit = result_storage.get_best_solution_fit()
    print("Status solver :", cp_model.status_solver)
    assert knapsack_problem.satisfy(sol)


if __name__ == "__main__":
    run_cpsat_knapsack()
