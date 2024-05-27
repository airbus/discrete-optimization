#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import platform

import pytest

from discrete_optimization.generic_tools.lp_tools import MilpSolverName, ParametersMilp
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file
from discrete_optimization.rcpsp.rcpsp_solution import PartialSolution, RCPSPSolution
from discrete_optimization.rcpsp.rcpsp_utils import (
    plot_resource_individual_gantt,
    plot_ressource_view,
)
from discrete_optimization.rcpsp.solver.rcpsp_lp_solver import LP_MRCPSP, LP_RCPSP

if platform.machine() == "arm64":
    pytest.skip(
        "Python-mip has issues with cbclib on macos arm64. "
        "See https://github.com/coin-or/python-mip/issues/167",
        allow_module_level=True,
    )


def test_rcpsp_sm_lp_cbc():
    files_available = get_data_available()
    file = [f for f in files_available if "j301_1.sm" in f][0]
    rcpsp_problem = parse_file(file)
    solver = LP_RCPSP(problem=rcpsp_problem, lp_solver=MilpSolverName.CBC)
    solver.init_model()
    results_storage: ResultStorage = solver.solve(
        parameters_milp=ParametersMilp.default()
    )
    solution, fit = results_storage.get_best_solution_fit()
    solution_rebuilt = RCPSPSolution(
        problem=rcpsp_problem, rcpsp_permutation=solution.rcpsp_permutation
    )
    fit_2 = rcpsp_problem.evaluate(solution_rebuilt)
    assert fit == -fit_2["makespan"]
    assert rcpsp_problem.satisfy(solution)
    plot_resource_individual_gantt(rcpsp_problem, solution)
    plot_ressource_view(rcpsp_problem, solution)


def test_rcpsp_mm_lp_cbc():
    files_available = get_data_available()
    file = [f for f in files_available if "j1010_1.mm" in f][0]
    rcpsp_problem = parse_file(file)
    rcpsp_problem.set_fixed_modes([1 for i in range(rcpsp_problem.n_jobs)])
    solver = LP_MRCPSP(problem=rcpsp_problem, lp_solver=MilpSolverName.CBC)
    solver.init_model(greedy_start=False)
    results_storage: ResultStorage = solver.solve(
        parameters_milp=ParametersMilp.default()
    )
    solution, fit = results_storage.get_best_solution_fit()
    solution_rebuilt = RCPSPSolution(
        problem=rcpsp_problem,
        rcpsp_modes=solution.rcpsp_modes,
        rcpsp_permutation=solution.rcpsp_permutation,
    )
    fit_2 = rcpsp_problem.evaluate(solution_rebuilt)
    assert rcpsp_problem.satisfy(solution)
    assert fit == -fit_2["makespan"]
    plot_resource_individual_gantt(rcpsp_problem, solution)
    plot_ressource_view(rcpsp_problem, solution)


def test_rcpsp_sm_lp_cbc_partial():
    files_available = get_data_available()
    file = [f for f in files_available if "j301_1.sm" in f][0]
    rcpsp_problem = parse_file(file)
    dummy_solution = rcpsp_problem.get_dummy_solution()
    some_constraints = {
        task: dummy_solution.rcpsp_schedule[task]["start_time"] for task in [1, 2, 3, 4]
    }
    partial_solution = PartialSolution(task_mode=None, start_times=some_constraints)
    partial_solution_for_lp = partial_solution
    solver = LP_MRCPSP(problem=rcpsp_problem, lp_solver=MilpSolverName.CBC)
    solver.init_model(partial_solution=partial_solution_for_lp, greedy_start=False)
    params_milp = ParametersMilp.default()
    params_milp.time_limit = 20
    store = solver.solve(parameters_milp=params_milp)
    solution, fit = store.get_best_solution_fit()
    solution: RCPSPSolution = solution  # just for autocompletion.
    for task in some_constraints:
        assert solution.get_start_time(task) == some_constraints[task]
    assert rcpsp_problem.satisfy(solution)
    plot_resource_individual_gantt(rcpsp_problem, solution)
    plot_ressource_view(rcpsp_problem, solution)


if __name__ == "__main__":
    test_rcpsp_sm_lp_cbc()
    # multi_mode_rcpsp_lp_grb()
    # single_mode_rcpsp_lp_cbc_robot()
    # single_mode_rcpsp_lp_grb()
    # single_mode_rcpsp_lp_lns()
    # multi_mode_rcpsp_cp()
