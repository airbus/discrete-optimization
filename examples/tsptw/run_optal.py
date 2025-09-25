#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import os

from discrete_optimization.datasets import get_data_home
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.tsptw.parser import get_data_available, parse_tsptw_file
from discrete_optimization.tsptw.problem import TSPTWProblem, TSPTWSolution
from discrete_optimization.tsptw.solvers.optal import OptalTspTwSolver

logging.basicConfig(level=logging.INFO)


def run_optal():
    data_available_ohlmann = get_data_available(
        data_folder=os.path.join(get_data_home(), "tsptw", "OhlmannThomas")
    )
    problem = parse_tsptw_file(data_available_ohlmann[2])
    solver = OptalTspTwSolver(problem=problem)
    solver.init_model(scaling_factor=100)
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    res = solver.solve(
        time_limit=100,
        parameters_cp=p,
        **{
            "worker0-1.searchType": "fdslb",
            "worker0-1.noOverlapPropagationLevel": 4,
            "worker0-1.cumulPropagationLevel": 3,
        },
    )
    sol = res.get_best_solution()
    print(problem.satisfy(sol), problem.evaluate(sol))


if __name__ == "__main__":
    run_optal()
