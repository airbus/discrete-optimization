#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

import pytest

from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.tsp.parser import get_data_available, parse_file
from discrete_optimization.tsp.problem import TspSolution
from discrete_optimization.tsp.solvers.dp import DpTspSolver, dp


@pytest.mark.parametrize("end_index", [0, 10])
def test_dp_solver(end_index):
    files = get_data_available()
    files = [f for f in files if "tsp_100_1" in f]
    model = parse_file(files[0], start_index=0, end_index=end_index)
    params_objective_function = get_default_objective_setup(problem=model)
    solver = DpTspSolver(model, params_objective_function=params_objective_function)
    solver.init_model()
    res = solver.solve(solver=dp.LNBS, time_limit=5)
    sol, fitness = res.get_best_solution_fit()
    sol: TspSolution
    assert model.satisfy(sol)
    assert sol.end_index == end_index
