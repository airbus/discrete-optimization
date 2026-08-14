#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import random

from discrete_optimization.rcpsp.problem import RcpspProblem, Task
from discrete_optimization.rcpsp_alternative.problem import (
    AlternativeSchedulingSubProblem,
    RcpspWithAlternativePath,
)


def create_problem_rcpsp(
    problem: RcpspProblem,
    nb_alternative_paths: int = 3,
    range_nb_subpath: tuple = (1, 4),
    range_len_subpath: tuple = (1, 5),
) -> RcpspWithAlternativePath:
    graph = problem.graph
    descendants = graph.descendants_map()
    ancestors = graph.ancestors_map()
    compatible_source_target: set[tuple[Task, Task]] = set()
    for t0 in problem.tasks_list:
        for t1 in problem.tasks_list:
            if t0 == t1:
                continue
            if t1 not in ancestors[t0] and t0 not in descendants[t1]:
                compatible_source_target.add((t0, t1))
    compatible_source_target = list(compatible_source_target)
    alternative_tasks = []
    alternative_tasks_data: dict[Task, dict[int, dict[str, int]]] = {}
    alternative_successors: dict[Task, list[Task]] = {}
    list_alternative_subproblem: list[AlternativeSchedulingSubProblem] = []
    all_durations = [
        problem.get_task_mode_duration(task, mode)
        for task in problem.tasks_list
        for mode in problem.get_task_modes(task)
    ]
    min_duration = min(all_durations)
    max_duration = max(all_durations)
    for i in range(nb_alternative_paths):
        source, sink = random.choice(compatible_source_target)
        nb_subpath = random.randint(range_nb_subpath[0], range_nb_subpath[1])
        list_paths = []
        for j in range(nb_subpath):
            path = []
            len_subpath = random.randint(range_len_subpath[0], range_len_subpath[1])
            for k in range(len_subpath):
                task_key = (
                    i,
                    j,
                    k,
                )  # I-th alternative subproblem, j-th subpath, k-th task in the subpath.
                alternative_tasks_data[task_key] = {
                    1: {"duration": random.randint(min_duration, max_duration)}
                }
                for r in problem.resources_list:
                    alternative_tasks_data[task_key][1][r] = random.randint(
                        0, problem.get_max_resource_capacity(r) // 2
                    )
                alternative_tasks.append(task_key)
                path.append(task_key)
            list_paths.append(path)
        list_alternative_subproblem.append(
            AlternativeSchedulingSubProblem(
                source_task=source,
                sink_task=sink,
                list_paths=list_paths,
                is_path_successors=True,
                nb_path_to_do=1,
            )
        )
    return RcpspWithAlternativePath(
        resources=problem.resources,
        non_renewable_resources=problem.non_renewable_resources,
        mode_details=problem.mode_details,
        successors=problem.successors,
        horizon=problem.horizon * 2,
        tasks_list=problem.tasks_list,
        source_task=problem.source_task,
        sink_task=problem.sink_task,
        name_task=problem.name_task,
        calendar_details=problem.calendar_details,
        special_constraints=problem.special_constraints,
        alternative_tasks=alternative_tasks,
        alternative_tasks_data=alternative_tasks_data,
        alternative_successors=alternative_successors,
        list_alternative_subproblem=list_alternative_subproblem,
    )
