#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.generic_rcpsp_tools.large_neighborhood_search_scheduling import (
    LargeNeighborhoodSearchScheduling,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file

logging.basicConfig(level=logging.INFO)


def example_lns_solver():
    files_available = get_data_available()
    file = [f for f in files_available if "j1201_1.sm" in f][0]
    rcpsp_problem: RCPSPModel = parse_file(file)
    solver = LargeNeighborhoodSearchScheduling(problem=rcpsp_problem)
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit_iter0 = 5
    parameters_cp.time_limit = 2
    results = solver.solve(
        nb_iteration_lns=100,
        skip_first_iteration=False,
        stop_first_iteration_if_optimal=False,
        parameters_cp=parameters_cp,
        nb_iteration_no_improvement=200,
        callbacks=[TimerStopper(total_seconds=100)],
    )
    sol, fit = results.get_best_solution_fit()
    assert rcpsp_problem.satisfy(sol)


if __name__ == "__main__":
    example_lns_solver()
