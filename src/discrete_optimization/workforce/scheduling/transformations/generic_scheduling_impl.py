#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from RCPSP to RCPSP Multiskill."""

from __future__ import annotations

import numpy as np

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
from discrete_optimization.workforce.scheduling.problem import (
    AllocSchedulingProblem,
    AllocSchedulingSolution,
    TasksDescription,
)
from discrete_optimization.workforce.scheduling.problem import (
    UnaryResource as WfUnaryResource,
)


def transform_solution_from_raw_generic_to_wf_sched(
    raw_sol: RawSolution[Task, UnaryResource, Skill], problem: AllocSchedulingProblem
) -> AllocSchedulingSolution:
    """Convert generic solution to workforce/scheduling solution.

    Args:
        solution:
        problem:

    Returns:

    """
    schedule = np.zeros((problem.number_tasks, 2), dtype=int)
    allocation = -np.ones(problem.number_tasks, dtype=int)
    for i_task in range(problem.number_tasks):
        task = problem.index_to_task[i_task]
        task_variable = raw_sol.task_variables[task]
        schedule[i_task, 0] = task_variable.start
        schedule[i_task, 1] = task_variable.end
        for team in task_variable.allocated:
            allocation[i_task] = problem.teams_to_index[team]
    return AllocSchedulingSolution(
        problem=problem, schedule=schedule, allocation=allocation
    )


class WfSchedulingToGenericSchedulingTransformation(
    ToGenericSchedulingImpl[AllocSchedulingProblem, AllocSchedulingSolution]
):
    """Transform RCPSP to GenericSchedulingImplProblem."""

    def transform_solution_from_raw_generic_to_specific(
        self,
        raw_sol: RawSolution[Task, UnaryResource, Skill],
        source_problem: AllocSchedulingProblem,
    ) -> AllocSchedulingSolution:
        return transform_solution_from_raw_generic_to_wf_sched(
            raw_sol=raw_sol, problem=source_problem
        )

    def get_forward_metadata(self) -> TransformationMetadata:
        return lossy_transformation(
            losses=[
                InformationLoss(
                    name="same_allocation",
                    loss_type=LossType.CONSTRAINT,
                    description="tasks to be performed by same team.",
                    impact=LossImpact.MAJOR,
                    reason="not possible with generic scheduling implementation.",
                ),
            ]
        )


class GenericSchedulingToWfSchedulingTransformation(
    FromGenericSchedulingImpl[AllocSchedulingProblem, AllocSchedulingSolution]
):
    """Transform GenericSchedulingImplProblem to RCPSP."""

    def get_forward_metadata(self) -> TransformationMetadata:
        return lossy_transformation(
            losses=[
                InformationLoss(
                    name="multimode",
                    loss_type=LossType.PARAMETER,
                    description="modes available for each task.",
                    impact=LossImpact.MAJOR,
                    reason="wf/scheduling is singlemode.",
                ),
                InformationLoss(
                    name="forbidden_intervals",
                    loss_type=LossType.CONSTRAINT,
                    description="constraints on fixed intervals with which a given task cannot overlap.",
                    impact=LossImpact.MAJOR,
                    reason="wf/scheduling cannot handle it.",
                ),
                InformationLoss(
                    name="no_overlap",
                    loss_type=LossType.CONSTRAINT,
                    description="constraints on tasks that cannot overlap.",
                    impact=LossImpact.MAJOR,
                    reason="wf/scheduling cannot handle it.",
                ),
                InformationLoss(
                    name="time_lags",
                    loss_type=LossType.CONSTRAINT,
                    description="time lags constraints between tasks.",
                    impact=LossImpact.MAJOR,
                    reason="wf/scheduling cannot handle time lags constraints.",
                ),
                InformationLoss(
                    name="skills",
                    loss_type=LossType.CONSTRAINT,
                    description="constraints on needed skills by tasks to be brought by unary resources.",
                    impact=LossImpact.MAJOR,
                    reason="wf/scheduling cannot handle it.",
                ),
                InformationLoss(
                    name="non_renewable_resources",
                    loss_type=LossType.CONSTRAINT,
                    description="constraints on needed non-renewable resource.",
                    impact=LossImpact.MAJOR,
                    reason="wf/scheduling cannot handle it.",
                ),
                InformationLoss(
                    name="objective",
                    loss_type=LossType.OBJECTIVE,
                    description="possibility to choose the objective",
                    impact=LossImpact.MODERATE,
                    reason="wf/scheduling has fixed objective.",
                ),
            ]
        )

    def transform_problem(
        self, source_problem: GenericSchedulingImplProblem
    ) -> AllocSchedulingProblem:
        """

        Args:
            source_problem:

        Returns:

        """
        team_names: list[WfUnaryResource] = source_problem.unary_resources_list
        calendar_team: dict[WfUnaryResource, list[tuple[int, int]]] = {
            unary_resource: [
                (start, end)
                for start, end, value in source_problem.get_resource_availabilities(
                    resource=unary_resource
                )
                if value > 0
            ]
            for unary_resource in source_problem.unary_resources_list
        }
        horizon: int = source_problem.horizon
        tasks_list = source_problem.tasks_list
        # should be singlemode, take first found mode for each task
        tasks_mode = {
            task: next(iter(source_problem.get_task_modes(task)))
            for task in source_problem.tasks_list
        }
        task_data: dict[Task, TasksDescription] = {
            task: TasksDescription(
                duration_task=source_problem.get_task_mode_duration(
                    task=task, mode=tasks_mode[task]
                ),
                resource_consumption={
                    str(resource): source_problem.get_cumulative_resource_consumption(
                        task=task, mode=tasks_mode[task], resource=resource
                    )
                    for resource in source_problem.cumulative_resources_list
                },
            )
            for task in source_problem.tasks_list
        }
        precedence_constraints: dict[Task, set[Task]] = {
            task: set(next_tasks)
            for task, next_tasks in source_problem.get_precedence_constraints().items()
        }
        available_team_for_activity: dict[Task, set[WfUnaryResource]] = {
            task: {
                unary_resource
                for unary_resource in source_problem.unary_resources_list
                if source_problem.is_compatible_task_unary_resource(
                    task=task, unary_resource=unary_resource
                )
            }
            for task in source_problem.tasks_list
        }
        start_window = {
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
        end_window = {
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
        # calendar lost
        resources_capacity: dict[str, int] = {
            str(resource): source_problem.get_resource_max_capacity(resource=resource)
            for resource in source_problem.non_skill_cumulative_resources_list
        }

        return AllocSchedulingProblem(
            team_names=team_names,
            calendar_team=calendar_team,
            horizon=horizon,
            tasks_list=tasks_list,
            tasks_data=task_data,
            same_allocation=[],
            precedence_constraints=precedence_constraints,
            available_team_for_activity=available_team_for_activity,
            start_window=start_window,
            end_window=end_window,
            resources_capacity=resources_capacity,
            original_start={},
            original_end={},
        )
