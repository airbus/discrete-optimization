#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.knapsack.knapsack_model import KnapsackSolution
from discrete_optimization.knapsack.knapsack_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.knapsack.solvers.knapsack_cpmpy import CPMPYKnapsackSolver


def test_knapsack_cpmyp():
    file = [f for f in get_data_available() if "ks_30_0" in f][0]
    knapsack_model = parse_file(file)
    solver = CPMPYKnapsackSolver(knapsack_model=knapsack_model)
    solver.init_model()
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 20
    res = solver.solve(solver="ortools", parameters_cp=parameters_cp)
    sol = res.get_best_solution()
    assert isinstance(sol, KnapsackSolution)
    assert knapsack_model.satisfy(sol)
