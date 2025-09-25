#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.singlemachine.parser import get_data_available, parse_file
from discrete_optimization.singlemachine.problem import WTSolution
from discrete_optimization.singlemachine.solvers.optal import OptalSingleMachineSolver

logging.basicConfig(level=logging.INFO)


def run_optal():
    problem = parse_file(get_data_available()[0])[5]
    solver = OptalSingleMachineSolver(problem)
    solver.init_model()
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    res = solver.solve(
        parameters_cp=p,
        time_limit=20,
        **{
            "worker0-1.searchType": "fdslb",
            "worker0-1.noOverlapPropagationLevel": 4,
            "worker0-1.cumulPropagationLevel": 3,
        },
    )
    sol: WTSolution = res.get_best_solution()
    print(problem.evaluate(sol))
    print(problem.satisfy(sol))


if __name__ == "__main__":
    run_optal()
