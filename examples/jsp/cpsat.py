#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.jsp.parser import get_data_available, parse_file
from discrete_optimization.jsp.solvers.cpsat import CpSatJspSolver

logging.basicConfig(level=logging.INFO)


def run_cpsat_jsp():
    file_path = get_data_available()[1]
    # file_path = [f for f in get_data_available() if "abz6" in f][0]
    problem = parse_file(file_path)
    print("File path ", file_path)
    print(
        "Problem with ",
        problem.n_jobs,
        " jobs, ",
        problem.n_all_jobs,
        " subjobs, and ",
        problem.n_machines,
        " machines",
    )
    solver = CpSatJspSolver(problem=problem)
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    res = solver.solve(parameters_cp=p, time_limit=10)
    sol = res.get_best_solution_fit()[0]
    print(solver.status_solver)
    assert problem.satisfy(sol)
    print(problem.evaluate(sol))


if __name__ == "__main__":
    run_cpsat_jsp()
