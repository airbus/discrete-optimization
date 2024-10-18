#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

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


def run_dp_fjsp_ws():
    files = fjsp_parser.get_data_available()
    file = [f for f in files if "Behnke4.fjs" in f][0]
    problem = fjsp_parser.parse_file(file)
    solver_ws = CpSatFjspSolver(problem=problem)
    g_sol = solver_ws.solve(time_limit=10)[-3][0]
    print(problem.evaluate(g_sol))
    solver = DpFjspSolver(problem=problem)
    solver.init_model(add_penalty_on_inefficiency=False)
    solver.set_warm_start(g_sol)
    res = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=20)],
        solver=dp.LNBS,
        time_limit=50,
        retrieve_intermediate_solutions=True,
    )
    print(g_sol.schedule)
    print(res[0][0].schedule)
    # assert g_sol.schedule == res[0][0].schedule
    print(res.get_best_solution_fit())
    print(problem.satisfy(res.get_best_solution_fit()[0]))
    print(problem.evaluate(res.get_best_solution_fit()[0]))


if __name__ == "__main__":
    # run_dp_fjsp()
    run_dp_fjsp_ws()
