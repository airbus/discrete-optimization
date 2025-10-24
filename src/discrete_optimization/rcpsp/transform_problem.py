#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  Transform a model without special constraints into one with those.
#  Also permits to pass from a classic RCPSP to a preemptive version
from discrete_optimization.rcpsp import SpecialConstraintsPreemptiveRcpspProblem
from discrete_optimization.rcpsp.problem import (
    RcpspProblem,
    SpecialConstraintsDescription,
)


def from_rcpsp_problem(
    rcpsp_problem: RcpspProblem,
    constraints: SpecialConstraintsDescription,
    preemptive=False,
) -> RcpspProblem:
    if preemptive:
        return SpecialConstraintsPreemptiveRcpspProblem(
            resources=rcpsp_problem.resources,
            non_renewable_resources=rcpsp_problem.non_renewable_resources,
            mode_details=rcpsp_problem.mode_details,
            successors=rcpsp_problem.successors,
            horizon=rcpsp_problem.horizon,
            special_constraints=constraints,
            tasks_list=rcpsp_problem.tasks_list,
            source_task=rcpsp_problem.source_task,
            sink_task=rcpsp_problem.sink_task,
            name_task=rcpsp_problem.name_task,
        )
    return RcpspProblem(
        resources=rcpsp_problem.resources,
        non_renewable_resources=rcpsp_problem.non_renewable_resources,
        mode_details=rcpsp_problem.mode_details,
        successors=rcpsp_problem.successors,
        horizon=rcpsp_problem.horizon,
        special_constraints=constraints,
        tasks_list=rcpsp_problem.tasks_list,
        source_task=rcpsp_problem.source_task,
        sink_task=rcpsp_problem.sink_task,
        name_task=rcpsp_problem.name_task,
    )
