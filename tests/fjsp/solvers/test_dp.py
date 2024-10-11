#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

import discrete_optimization.fjsp.parser as fjsp_parser
import discrete_optimization.jsp.parser as jsp_parser
from discrete_optimization.fjsp.problem import FJobShopProblem, Job
from discrete_optimization.fjsp.solvers.dp import DpFjspSolver, dp
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


def test_cpsat_fjsp():
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
