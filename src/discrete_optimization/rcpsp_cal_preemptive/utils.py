#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp_cal_preemptive.problem import (
    CalendarPreemptiveRcpspProblem,
)


def load_calendar_preemptive_rcpsp_problem(
    problem: RcpspProblem = None, frequency: int = 5
) -> CalendarPreemptiveRcpspProblem:
    # file = get_data_available()[1]
    if problem is None:
        file = [f for f in get_data_available() if "j1201_1" in f][0]
        problem = parse_file(file)
    for r in problem.resources_list:
        if r not in problem.non_renewable_resources:
            max_capa = problem.get_max_resource_capacity(r)
            problem.resources[r] = [max_capa] * (problem.horizon * 3)
            if True:
                for i in range(len(problem.resources[r])):
                    if i % frequency == 0:
                        problem.resources[r][i] = 0
                    if i % frequency == 1:
                        problem.resources[r][i] = max_capa - 1
        else:
            max_capa = problem.get_max_resource_capacity(r)
            problem.resources[r] = [max_capa] * (problem.horizon * 3)
    problem.horizon = problem.horizon * 3
    problem.update_problem()
    preemptive = CalendarPreemptiveRcpspProblem(
        resources=problem.resources,
        non_renewable_resources=problem.non_renewable_resources,
        mode_details=problem.mode_details,
        successors=problem.successors,
        horizon=problem.horizon,
        tasks_list=problem.tasks_list,
        source_task=problem.source_task,
        sink_task=problem.sink_task,
        special_constraints=problem.special_constraints,
        calendar_preemptive_tasks=set(problem.tasks_list),
    )
    return preemptive
