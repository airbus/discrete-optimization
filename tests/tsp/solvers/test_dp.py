#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

import pytest

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
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


def test_dp_solver_ws():
    files = get_data_available()
    files = [f for f in files if "tsp_100_1" in f]
    model = parse_file(files[0], start_index=0, end_index=0)
    params_objective_function = get_default_objective_setup(problem=model)
    from discrete_optimization.tsp.solvers.ortools_routing import ORtoolsTspSolver

    solver_ws = ORtoolsTspSolver(model)
    sol = solver_ws.solve(time_limit=5)[-1][0]
    solver = DpTspSolver(model, params_objective_function=params_objective_function)
    solver.init_model()
    solver.set_warm_start(sol)
    res = solver.solve(
        solver=dp.LNBS,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        retrieve_intermediate_solutions=True,
        threads=5,
        time_limit=40,
    )
    sol_, fitness = res.get_best_solution_fit()
    assert model.satisfy(sol_)
    assert res[0][0].permutation == sol.permutation
