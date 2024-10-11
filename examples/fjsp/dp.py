#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

import discrete_optimization.fjsp.parser as fjsp_parser
import discrete_optimization.jsp.parser as jsp_parser
from discrete_optimization.fjsp.problem import FJobShopProblem, Job
from discrete_optimization.fjsp.solvers.dp import DpFjspSolver, dp
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.jsp.problem import JobShopProblem, Subjob

logging.basicConfig(level=logging.INFO)


def run_dp_jsp():
    file_path = jsp_parser.get_data_available()[1]
    file_path = [f for f in jsp_parser.get_data_available() if "ta68" in f][0]
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
    res = solver.solve(solver=dp.LNBS, time_limit=20)
    print(res.get_best_solution_fit())
    print(problem.satisfy(res.get_best_solution_fit()[0]))


def run_dp_fjsp():
    files = fjsp_parser.get_data_available()
    print(files)
    file = [f for f in files if "Behnke46.fjs" in f][0]
    print(file)
    problem = fjsp_parser.parse_file(file)
    print(problem)
    solver = DpFjspSolver(problem=problem)
    solver.init_model(add_penalty_on_inefficiency=False)
    res = solver.solve(solver=dp.LNBS, time_limit=50)
    print(res.get_best_solution_fit())
    print(problem.satisfy(res.get_best_solution_fit()[0]))


if __name__ == "__main__":
    run_dp_fjsp()
