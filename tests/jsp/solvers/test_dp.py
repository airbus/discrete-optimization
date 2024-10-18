#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import didppy as dp

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.jsp.parser import get_data_available, parse_file
from discrete_optimization.jsp.solvers.cpsat import CpSatJspSolver
from discrete_optimization.jsp.solvers.dp import DpJspSolver


def test_dp_jsp():
    problem = parse_file(get_data_available()[0])
    solver = DpJspSolver(problem=problem)
    res = solver.solve(solver=dp.LNBS, time_limit=5)
    assert problem.satisfy(res.get_best_solution_fit()[0])


def test_dp_jsp_ws():
    # file_path = get_data_available()[1]
    file_path = [f for f in get_data_available() if "ta68" in f][0]
    problem = parse_file(file_path)
    solver_ws = CpSatJspSolver(problem)
    sol_ws = solver_ws.solve(time_limit=2)[0][0]
    solver = DpJspSolver(problem=problem)
    solver.init_model()
    solver.set_warm_start(sol_ws)
    res = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        retrieve_intermediate_solutions=True,
        solver=dp.LNBS,
        time_limit=100,
    )
    sol = res.get_best_solution_fit()[0]
    assert problem.satisfy(sol)
