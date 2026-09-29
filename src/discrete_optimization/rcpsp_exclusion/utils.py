#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import random
from typing import Hashable

from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp_exclusion.problem import (
    RcpspProblemWithExclusion,
)


def create_exclusion_rcpsp_problem(
    problem: RcpspProblem = None,
    nb_exclusion_resource: int = 1,
    proportion_blocking_tasks: float = 0.1,
    proportion_blocked_tasks: float = 0.1,
    range_capacities: tuple[int, int] = (1, 1),
) -> RcpspProblemWithExclusion:
    # file = get_data_available()[1]
    if problem is None:
        file = [f for f in get_data_available() if "j1201_1" in f][0]
        problem = parse_file(file)
    exclusion_resources = [f"Z{i}" for i in range(nb_exclusion_resource)]
    exclusion_resource_capacity: dict[str, int] = {
        z: random.randint(*range_capacities) for z in exclusion_resources
    }
    exclusion_resource_consumptions: dict[Hashable, dict[int, dict[str, int]]] = {}
    exclusion_resource_boolean: dict[Hashable, dict[int, dict[str, bool]]] = {}
    for z in exclusion_resources:
        blocking_tasks = random.sample(
            problem.tasks_list, int(proportion_blocking_tasks * problem.n_jobs)
        )
        blocked_tasks = random.sample(
            [t for t in problem.tasks_list if t not in blocking_tasks],
            int(proportion_blocked_tasks * problem.n_jobs),
        )
        for t in blocking_tasks:
            if t not in exclusion_resource_boolean:
                exclusion_resource_boolean[t] = {
                    m: {} for m in problem.get_task_modes(t)
                }
            for m in exclusion_resource_boolean[t]:
                exclusion_resource_boolean[t][m][z] = True
        for t in blocked_tasks:
            if t not in exclusion_resource_consumptions:
                exclusion_resource_consumptions[t] = {
                    m: {} for m in problem.get_task_modes(t)
                }
            for m in exclusion_resource_consumptions[t]:
                exclusion_resource_consumptions[t][m][z] = 1
    problem.horizon = problem.horizon * 3
    problem.update_problem()
    preemptive = RcpspProblemWithExclusion(
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
        exclusion_resource_capacity=exclusion_resource_capacity,
        exclusion_resource_consumptions=exclusion_resource_consumptions,
        exclusion_resource_boolean=exclusion_resource_boolean,
    )
    return preemptive
