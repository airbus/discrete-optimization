#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import os

from discrete_optimization.tsptw.parser import (
    get_data_available,
    get_data_home,
    parse_tsptw_file,
)
from discrete_optimization.tsptw.problem import TSPTWProblem, TSPTWSolution
from discrete_optimization.tsptw.solvers.dp import DpTspTwSolver, dp

logging.basicConfig(level=logging.INFO)


def run_dp():
    data_available_ohlmann = get_data_available(
        data_folder=os.path.join(get_data_home(), "tsptw", "OhlmannThomas")
    )
    problem = parse_tsptw_file(data_available_ohlmann[2])
    solver = DpTspTwSolver(problem=problem)
    solver.init_model(add_dominated_transition=False)
    res = solver.solve(
        time_limit=10, retrieve_intermediate_solutions=False, threads=10, solver=dp.CABS
    )
    sol = res.get_best_solution()
    print(problem.satisfy(sol), problem.evaluate(sol))
    # INFO:discrete_optimization.generic_tools.dyn_prog_tools:Objective = 455.03149999999994, False
    # INFO:discrete_optimization.generic_tools.dyn_prog_tools:Bound = 427.11990000000003
    # INFO:discrete_optimization.generic_tools.dyn_prog_tools:Is optimal False
    # INFO:discrete_optimization.generic_tools.dyn_prog_tools:Is infeasible False


if __name__ == "__main__":
    run_dp()
