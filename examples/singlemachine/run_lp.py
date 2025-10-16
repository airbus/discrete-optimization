#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.lp_tools import (
    OrtoolsMathOptMilpSolver,
    mathopt,
)
from discrete_optimization.singlemachine.parser import get_data_available, parse_file
from discrete_optimization.singlemachine.problem import WTSolution
from discrete_optimization.singlemachine.solvers.lp import MathOptSingleMachineSolver

logging.basicConfig(level=logging.INFO)


def run_lp():
    problem = parse_file(get_data_available()[0])[0]
    solver = MathOptSingleMachineSolver(problem)
    solver.init_model()
    res = solver.solve(
        time_limit=10,
        mathopt_solver_type=mathopt.SolverType.CP_SAT,
        mathopt_enable_output=True,
    )
    sol: WTSolution = res.get_best_solution()
    print(problem.evaluate(sol))
    print(problem.satisfy(sol))


if __name__ == "__main__":
    run_lp()
