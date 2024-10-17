#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

import discrete_optimization.fjsp.parser as fjsp_parser
import discrete_optimization.jsp.parser as jsp_parser
from discrete_optimization.fjsp.problem import Job
from discrete_optimization.fjsp.solvers.cpsat import CpSatFjspSolver, FJobShopProblem
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.jsp.problem import JobShopProblem, Subjob

logging.basicConfig(level=logging.INFO)


def run_cpsat_jsp():
    file_path = jsp_parser.get_data_available()[1]
    # file_path = [f for f in get_data_available() if "abz6" in f][0]
    problem: JobShopProblem = jsp_parser.parse_file(file_path)
    fproblem = FJobShopProblem(
        list_jobs=[
            Job(job_id=i, sub_jobs=[[sj] for sj in problem.list_jobs[i]])
            for i in range(problem.n_jobs)
        ],
        n_jobs=problem.n_jobs,
        n_machines=problem.n_machines,
    )
    solver = CpSatFjspSolver(problem=fproblem)
    p = ParametersCp.default_cpsat()
    p.nb_process = 12
    res = solver.solve(parameters_cp=p, time_limit=20)
    print(res.get_best_solution_fit())


def run_cpsat_fjsp():
    files = fjsp_parser.get_data_available()
    print(files)
    file = [f for f in files if "Behnke60.fjs" in f][0]
    print(file)
    problem = fjsp_parser.parse_file(file)
    print(problem)
    solver = CpSatFjspSolver(problem=problem)
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    res = solver.solve(
        parameters_cp=p,
        time_limit=300,
        ortools_cpsat_solver_kwargs=dict(log_search_progress=True),
        duplicate_temporal_var=True,
        add_cumulative_constraint=True,
    )
    print(solver.get_status_solver())
    print(res.get_best_solution_fit())


if __name__ == "__main__":
    run_cpsat_fjsp()
