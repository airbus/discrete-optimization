#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from/to generic scheduling to/from RCPSP."""

from __future__ import annotations

import itertools
from collections.abc import Hashable

from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    GenericSchedulingImplProblem,
    Skill,
    Task,
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import (
    RawSolution,
)
from discrete_optimization.generic_tasks_tools.transformations.generic_scheduling_impl import (
    FromGenericSchedulingImpl,
    ToGenericSchedulingImpl,
)
from discrete_optimization.generic_tools.transformation import (
    InformationLoss,
    LossImpact,
    LossType,
    TransformationMetadata,
    lossy_transformation,
)
from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp.solution import RcpspSolution
from discrete_optimization.rcpsp.special_constraints import (
    SpecialConstraintsDescription,
)


def transform_solution_from_raw_generic_to_rcpsp(
    raw_sol: RawSolution[Task, UnaryResource, Skill], problem: RcpspProblem
) -> RcpspSolution:
    """Convert generic solution to RCPSP solution.

    Args:
        solution:
        problem:

    Returns:

    """
    schedule = {}
    modes_dict = {}
    for task, task_variable in raw_sol.task_variables.items():
        schedule[task] = {
            "start_time": task_variable.start,
            "end_time": task_variable.end,
        }
        modes_dict[task] = task_variable.mode
    return RcpspSolution(
        problem=problem,
        rcpsp_schedule=schedule,
        rcpsp_modes=[modes_dict[t] for t in problem.tasks_list_non_dummy],
    )


class RcpspToGenericSchedulingTransformation(
    ToGenericSchedulingImpl[RcpspProblem, RcpspSolution]
):
    """Transform RCPSP to GenericSchedulingImplProblem."""

    def transform_solution_from_raw_generic_to_specific(
        self,
        raw_sol: RawSolution[Task, UnaryResource, Skill],
        source_problem: RcpspProblem,
    ) -> RcpspSolution:
        return transform_solution_from_raw_generic_to_rcpsp(
            raw_sol=raw_sol, problem=source_problem
        )

    def get_forward_metadata(self) -> TransformationMetadata:
        return lossy_transformation(
            losses=[
                InformationLoss(
                    name="task_mode",
                    loss_type=LossType.CONSTRAINT,
                    description="fixed modes by tasks in special constraints.",
                    impact=LossImpact.MAJOR,
                    reason="not possible with generic scheduling implementation.",
                ),
                InformationLoss(
                    name="pair_mode_constraint",
                    loss_type=LossType.CONSTRAINT,
                    description="constraints on pair of tasks that must share a mode or a score.",
                    impact=LossImpact.MAJOR,
                    reason="not possible with generic scheduling implementation.",
                ),
            ]
        )


class GenericSchedulingToRcpspTransformation(
    FromGenericSchedulingImpl[RcpspProblem, RcpspSolution]
):
    """Transform GenericSchedulingImplProblem to RCPSP."""

    def get_forward_metadata(self) -> TransformationMetadata:
        return lossy_transformation(
            losses=[
                InformationLoss(
                    name="forbidden_intervals",
                    loss_type=LossType.CONSTRAINT,
                    description="constraints on fixed intervals with which a given task cannot overlap.",
                    impact=LossImpact.MAJOR,
                    reason="rcpsp special constraints cannot handle it.",
                ),
                InformationLoss(
                    name="time_lags",
                    loss_type=LossType.CONSTRAINT,
                    description="time lags constraints, but only end_to_start max, start_to_end min, end_to_end min/max.",
                    impact=LossImpact.MAJOR,
                    reason="rcpsp special constraints can only handle the other time lags constraints.",
                ),
                InformationLoss(
                    name="allocated",
                    loss_type=LossType.PARAMETER,
                    description="unary resources allocated to each task",
                    reason="rcpsp does not model allocation",
                    impact=LossImpact.MAJOR,
                ),
                InformationLoss(
                    name="objective",
                    loss_type=LossType.OBJECTIVE,
                    description="whenever the objective is not just makespan",
                    impact=LossImpact.MODERATE,
                    reason="rcpsp has fixed objective.",
                ),
            ]
        )

    def transform_problem(
        self, source_problem: GenericSchedulingImplProblem
    ) -> RcpspProblem:
        """

        Args:
            source_problem:

        Returns:

        """
        horizon = source_problem.horizon
        resources = {
            str(resource): source_problem.get_non_renewable_resource_capacity(
                resource=resource
            )
            for resource in source_problem.non_renewable_resources_list
        } | {
            str(resource): source_problem.get_resource_calendar(resource=resource)
            for resource in source_problem.non_skill_cumulative_resources_list
        }
        non_renewable_resources = [
            str(resource) for resource in source_problem.non_renewable_resources_list
        ]
        mode_details: dict[Hashable, dict[int, dict[str, int]]] = {
            task: {
                mode: {
                    str(
                        resource
                    ): source_problem.get_non_renewable_resource_consumption(
                        resource=resource, task=task, mode=mode
                    )
                    for resource in source_problem.non_renewable_resources_list
                }
                | {
                    str(resource): source_problem.get_cumulative_resource_consumption(
                        resource=resource, task=task, mode=mode
                    )
                    for resource in source_problem.cumulative_resources_list
                }
                | {
                    "duration": source_problem.get_task_mode_duration(
                        task=task, mode=mode
                    )
                }
                for mode in source_problem.get_task_modes(task)
            }
            for task in source_problem.tasks_list
        }
        successors = {
            task: list(next_tasks)
            for task, next_tasks in source_problem.get_precedence_constraints().items()
        }
        # find source task and sink task or create them
        precedence_graph = source_problem.get_precedence_graph()
        first_tasks = [
            task
            for task, n_predecessors in precedence_graph.graph_nx.in_degree
            if n_predecessors == 0
        ]
        if len(first_tasks) > 1:
            # add a source task
            source_task = "source"
            if source_task in mode_details or source_task in successors:
                raise ValueError(
                    f"The source problem does not have a source task but has a task named '{source_task}'."
                )
            mode_details[source_task] = {1: {"duration": 0}}
            successors[source_task] = first_tasks
        else:
            source_task = first_tasks[0]
        last_tasks = [
            task
            for task, n_successors in precedence_graph.graph_nx.out_degree
            if n_successors == 0
        ]
        if len(last_tasks) > 1:
            # add a sink task
            sink_task = "sink"
            if sink_task in mode_details or sink_task in successors:
                raise ValueError(
                    f"The source problem does not have a sink task but has a task named '{sink_task}'."
                )
            mode_details[sink_task] = {1: {"duration": 0}}
            for task in last_tasks:
                successors[task] = [sink_task]
        else:
            sink_task = last_tasks[0]
        # special constraints
        # time windows
        start_times_window = {
            task: (
                source_problem.get_task_start_or_end_lower_bound(
                    task=task, start_or_end=StartOrEnd.START
                ),
                source_problem.get_task_start_or_end_upper_bound(
                    task=task, start_or_end=StartOrEnd.START
                ),
            )
            for task in source_problem.tasks_list
        }
        end_times_window = {
            task: (
                source_problem.get_task_start_or_end_lower_bound(
                    task=task, start_or_end=StartOrEnd.END
                ),
                source_problem.get_task_start_or_end_upper_bound(
                    task=task, start_or_end=StartOrEnd.END
                ),
            )
            for task in source_problem.tasks_list
        }
        # time lags
        start_after_end_plus_offset = (
            source_problem.get_end_to_start_min_time_lags()
            + [
                (t2, t1, -offset)
                for t1, t2, offset in source_problem.get_start_to_end_max_time_lags()
            ]
        )
        start_to_start_min_time_lag = source_problem.get_start_to_start_min_time_lags()
        start_to_start_max_time_lag = source_problem.get_start_to_start_max_time_lags()
        # loss: end_to_start max, start_to_end min, end_to_end min/max

        # disjunctive tasks
        disjunctive_tasks: list[tuple[Task, Task]] = []
        for tasks in source_problem.get_no_overlap():
            if len(tasks) >= 2:
                disjunctive_tasks.extend(itertools.combinations(tasks, 2))

        special_constraints = SpecialConstraintsDescription(
            start_times_window=start_times_window,
            end_times_window=end_times_window,
            start_after_end_plus_offset=start_after_end_plus_offset,
            start_to_start_min_time_lag=start_to_start_min_time_lag,
            start_to_start_max_time_lag=start_to_start_max_time_lag,
            disjunctive_tasks=disjunctive_tasks,
        )
        return RcpspProblem(
            resources=resources,
            non_renewable_resources=non_renewable_resources,
            mode_details=mode_details,
            successors=successors,
            horizon=horizon,
            source_task=source_task,
            sink_task=sink_task,
            special_constraints=special_constraints,
        )
