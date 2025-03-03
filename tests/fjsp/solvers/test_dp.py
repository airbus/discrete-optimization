#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

import pytest

import discrete_optimization.fjsp.parser as fjsp_parser
import discrete_optimization.jsp.parser as jsp_parser
from discrete_optimization.fjsp.problem import FJobShopProblem, Job
from discrete_optimization.fjsp.solvers.cpsat import CpSatFjspSolver
from discrete_optimization.fjsp.solvers.dp import DpFjspSolver, dp
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.jsp.problem import JobShopProblem

logging.basicConfig(level=logging.INFO)


def test_fjsp_solver_on_jsp():
    file_path = jsp_parser.get_data_available()[1]
    problem: JobShopProblem = jsp_parser.parse_file(file_path)
    fproblem = FJobShopProblem(
        list_jobs=[
            Job(job_id=i, sub_jobs=[[sj] for sj in problem.list_jobs[i]])
            for i in range(problem.n_jobs)
        ],
        n_jobs=problem.n_jobs,
        n_machines=problem.n_machines,
    )
    solver = DpFjspSolver(problem=fproblem)
    res = solver.solve(solver=dp.LNBS, time_limit=10)
    sol, _ = res.get_best_solution_fit()
    assert fproblem.satisfy(sol)


@pytest.mark.skip("fjsp datasets temporary not available.")
def test_dp_fjsp():
    files = fjsp_parser.get_data_available()
    file = [f for f in files if "Behnke60.fjs" in f][0]
    problem = fjsp_parser.parse_file(file)
    solver = DpFjspSolver(problem=problem)
    res = solver.solve(
        solver=dp.LNBS,
        time_limit=10,
    )
    sol, _ = res.get_best_solution_fit()
    assert problem.satisfy(sol)


@pytest.mark.skip("fjsp datasets temporary not available.")
def test_dp_fjsp_ws():
    files = fjsp_parser.get_data_available()
    file = [f for f in files if "Behnke4.fjs" in f][0]
    problem = fjsp_parser.parse_file(file)
    solver_ws = CpSatFjspSolver(problem=problem)
    g_sol = solver_ws.solve(time_limit=10)[0][0]
    solver = DpFjspSolver(problem=problem)
    solver.init_model(add_penalty_on_inefficiency=False)
    solver.set_warm_start(g_sol)
    res = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        solver=dp.LNBS,
        time_limit=3,
        retrieve_intermediate_solutions=True,
    )
    assert problem.satisfy(res[0][0])
