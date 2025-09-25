#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.fjsp.parser import get_data_available, parse_file
from discrete_optimization.fjsp.solvers.optal import OptalFJspSolver
from discrete_optimization.generic_tools.cp_tools import ParametersCp

logging.basicConfig(level=logging.INFO)


def run_optal():
    file = get_data_available()[0]
    problem = parse_file(file)
    p = ParametersCp.default_cpsat()
    solver = OptalFJspSolver(problem=problem)
    solver.init_model()
    res = solver.solve(parameters_cp=p, time_limit=5)
    sol = res.get_best_solution()
    print(solver.status_solver)
    print(problem.satisfy(sol), problem.evaluate(sol))


if __name__ == "__main__":
    run_optal()
