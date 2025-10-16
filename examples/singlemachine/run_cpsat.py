#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.singlemachine.parser import get_data_available, parse_file
from discrete_optimization.singlemachine.solvers.cpsat import CpsatWTSolver, WTSolution

logging.basicConfig(level=logging.INFO)


def run_cpsat():
    problem = parse_file(get_data_available()[0])[0]
    solver = CpsatWTSolver(problem)
    solver.init_model()
    res = solver.solve(
        time_limit=10,
        parameters_cp=ParametersCp.default_cpsat(),
        callbacks=[
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
    )
    sol: WTSolution = res.get_best_solution()
    print(problem.evaluate(sol))
    print(problem.satisfy(sol))


if __name__ == "__main__":
    run_cpsat()
