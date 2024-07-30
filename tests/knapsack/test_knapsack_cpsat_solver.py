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


def test_cp_knapsack():
    file = [f for f in get_data_available() if "ks_30_0" in f][0]
    knapsack_model = parse_file(file)
    cp_model = CPSatKnapsackSolver(knapsack_model)
    cp_model.init_model()
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 10
    result_storage = cp_model.solve(parameters_cp=parameters_cp)
    sol, fit = result_storage.get_best_solution_fit()
    assert knapsack_model.satisfy(sol)

    start_solution = knapsack_model.get_dummy_solution()

    # first solution is not start_solution
    assert result_storage[0][0].list_taken != start_solution.list_taken

    # warm start at first solution
    cp_model.set_warm_start(start_solution)
    # force first solution to be the hinted one
    result_storage = cp_model.solve(
        parameters_cp=parameters_cp,
        ortools_cpsat_solver_kwargs=dict(fix_variables_to_their_hinted_value=True),
    )
    assert result_storage[0][0].list_taken == start_solution.list_taken
