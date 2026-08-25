#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Define the usecases on which we want non-regression.

(We will store previous results and compare newer versions of solvers to them)

"""

import numpy as np
from pytest_cases import parametrize

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_problem import ModeOptim
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp.solution import RcpspSolution
from discrete_optimization.rcpsp.solvers.cpsat import (
    CpSatCumulativeResourceRcpspSolver,
    CpSatRcpspSolver,
    CpSatResourceRcpspSolver,
)
from discrete_optimization.rcpsp.special_constraints import (
    SpecialConstraintsDescription,
)

# Problems


def parse_rcpsp_problem(model):
    files_available = get_data_available()
    file = [f for f in files_available if model in f][0]
    return parse_file(file)


@parametrize("model", ["j301_1.sm", "j1010_1.mm"])
def problem_simple(model):
    return parse_rcpsp_problem(model)


@parametrize("model", ["j301_1.sm", "j1010_1.mm"])
def problem_with_calendar_resource(model):
    rcpsp_problem = parse_rcpsp_problem(model)
    for resource in rcpsp_problem.resources:
        rcpsp_problem.resources[resource] = np.array(
            rcpsp_problem.get_resource_availability_array(resource)
        )
        rcpsp_problem.resources[resource][10:15] = 0
    rcpsp_problem.update_problem()
    return rcpsp_problem


mode_details = {
    1: {1: {"duration": 0}},  # dummy start
    2: {1: {"duration": 3, "R1": 1}},
    3: {1: {"duration": 2, "R1": 1}},
    4: {1: {"duration": 4, "R1": 1}},
    5: {1: {"duration": 0}},  # dummy end
}

successors = {
    1: [2, 3],
    2: [5],
    3: [4],
    4: [5],
    5: [],
}

resources = {"R1": 2}


def problem_special_constraints_start_together():
    return RcpspProblem(
        resources=resources,
        non_renewable_resources=[],
        mode_details=mode_details,
        successors=successors,
        horizon=100,
        special_constraints=SpecialConstraintsDescription(
            start_together=[(2, 3)],  # tasks 2 and 3 start together
        ),
    )


def problem_special_constraints_start_at_end():
    return RcpspProblem(
        resources=resources,
        non_renewable_resources=[],
        mode_details=mode_details,
        successors=successors,
        horizon=100,
        special_constraints=SpecialConstraintsDescription(
            start_at_end=[(3, 4)],  # task 4 starts when task 3 ends
        ),
    )


def problem_special_constraints_start_times_window():
    return RcpspProblem(
        resources=resources,
        non_renewable_resources=[],
        mode_details=mode_details,
        successors=successors,
        horizon=100,
        special_constraints=SpecialConstraintsDescription(
            start_times_window={2: (5, 10)},  # task 2 must start between 5 and 10
        ),
    )


def problem_special_constraints_disjunctive_tasks():
    return RcpspProblem(
        resources=resources,
        non_renewable_resources=[],
        mode_details=mode_details,
        successors=successors,
        horizon=100,
        special_constraints=SpecialConstraintsDescription(
            disjunctive_tasks=[(2, 3)],  # tasks 2 and 3 cannot overlap
        ),
    )


def problem_special_constraints_start_to_start_min_time_lag_negative_offset():
    mode_details = {
        1: {1: {"duration": 0}},  # dummy start
        2: {1: {"duration": 5, "R1": 1}},
        3: {1: {"duration": 3, "R1": 1}},
        4: {1: {"duration": 2, "R1": 1}},
        5: {1: {"duration": 0}},  # dummy end
    }
    successors = {
        1: [2, 3, 4],
        2: [5],
        3: [5],
        4: [5],
        5: [],
    }
    resources = {"R1": 2}
    special_constraints = SpecialConstraintsDescription(
        start_to_start_min_time_lag=[(2, 3, -3)],
    )
    return RcpspProblem(
        resources=resources,
        non_renewable_resources=[],
        mode_details=mode_details,
        successors=successors,
        horizon=100,
        special_constraints=special_constraints,
    )


def problem_special_constraints_start_to_start_min_time_lag_positive_offset():
    mode_details = {
        1: {1: {"duration": 0}},  # dummy start
        2: {1: {"duration": 2, "R1": 1}},
        3: {1: {"duration": 3, "R1": 1}},
        4: {1: {"duration": 0}},  # dummy end
    }
    successors = {
        1: [2, 3],
        2: [4],
        3: [4],
        4: [],
    }
    resources = {"R1": 1}
    # Test with positive offset: task 3 must start at least 5 units after task 2 starts
    # Constraint: start(2) + 5 <= start(3)
    special_constraints = SpecialConstraintsDescription(
        start_to_start_min_time_lag=[(2, 3, 5)],
    )
    return RcpspProblem(
        resources=resources,
        non_renewable_resources=[],
        mode_details=mode_details,
        successors=successors,
        horizon=100,
        special_constraints=special_constraints,
    )


# Solvers


@parametrize("time_limit", [30])
def solver_cpsat(time_limit):
    solver_cls = CpSatRcpspSolver
    parameters_cp = (
        ParametersCp.default()
    )  # only 1 process to avoid discrepancy with github runners
    kwargs = dict(time_limit=time_limit, parameters_cp=parameters_cp)

    def objective_fn(solution: RcpspSolution):
        return solution.get_max_end_time()

    mode_optim = ModeOptim.MINIMIZATION
    return solver_cls, kwargs, objective_fn, mode_optim


@parametrize("time_limit", [30])
def solver_cpsat_resource(time_limit):
    solver_cls = CpSatResourceRcpspSolver
    parameters_cp = (
        ParametersCp.default()
    )  # only 1 process to avoid discrepancy with github runners
    kwargs = dict(time_limit=time_limit, parameters_cp=parameters_cp)

    def objective_fn(solution: RcpspSolution):
        return (
            solution.get_max_end_time(),
            solution.compute_nb_calendar_resources_used()
            + solution.compute_nb_non_renewable_resources_used(),
        )

    mode_optim = ModeOptim.MINIMIZATION

    return solver_cls, kwargs, objective_fn, mode_optim


@parametrize("time_limit", [30])
def solver_cpsat_cumulative_resource(time_limit):
    solver_cls = CpSatCumulativeResourceRcpspSolver
    parameters_cp = (
        ParametersCp.default()
    )  # only 1 process to avoid discrepancy with github runners
    kwargs = dict(time_limit=time_limit, parameters_cp=parameters_cp)

    def objective_fn(solution: RcpspSolution):
        return (
            solution.get_max_end_time(),
            solution.compute_aggregated_calendar_resources_levels()
            + solution.compute_aggregated_non_renewable_resources_consumptions(),
        )

    mode_optim = ModeOptim.MINIMIZATION

    return solver_cls, kwargs, objective_fn, mode_optim
