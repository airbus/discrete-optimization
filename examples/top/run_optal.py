#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.generic_tools.callbacks.loggers import ProblemEvaluateLogger
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.top.parser import get_data_available, parse_file
from discrete_optimization.top.solvers.optal import OptalTopSolver

logging.basicConfig(level=logging.INFO)


def run_optal():
    files, files_dict = get_data_available()
    print(files)
    print(files[0])
    problem = parse_file(files[4])
    solver = OptalTopSolver(problem)
    solver.init_model(scaling=100)
    res = solver.solve(
        parameters_cp=ParametersCp.default_cpsat(),
        callbacks=[
            ProblemEvaluateLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
        time_limit=30,
    )
    sol = res[-1][0]
    print(problem.satisfy(sol), problem.evaluate(sol))


if __name__ == "__main__":
    run()
