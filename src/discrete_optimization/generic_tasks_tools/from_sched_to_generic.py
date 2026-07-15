#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from GenericSchedulingProblems to GenericSchedulingProblemImpl
This can be shared by all rcpsp/shop problem to generate the corresponding GenericSchedulingProblemImpl,
just need to do the solution translation.
"""

from typing import Callable, Iterable, Optional

from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.generic_scheduling import (
    GenericSchedulingProblem,
    GenericSchedulingSolution,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    AnyResource,
    GenericSchedulingImplProblem,
    GenericSchedulingImplSolution,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import Objective
from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)


class SchedulingToGenericTransformation(
    ProblemTransformation[
        GenericSchedulingProblem,
        GenericSchedulingSolution,
        GenericSchedulingImplProblem,
        GenericSchedulingImplSolution,
    ]
):
    def __init__(
        self,
        objective: Objective | Iterable[tuple[Objective, int]] = Objective.MAKESPAN,
        custom_evaluate_fn: Optional[
            Callable[[GenericSchedulingImplSolution], int]
        ] = None,
        objective_resource_weights: Optional[dict[AnyResource, int]] = None,
    ):
        self.objective = objective
        self.custom_evaluate_fn = custom_evaluate_fn
        self.objective_resource_weights = objective_resource_weights

    def transform_problem(
        self, source_problem: GenericSchedulingProblem
    ) -> GenericSchedulingImplProblem:
        return GenericSchedulingImplProblem(
            horizon=source_problem.get_makespan_upper_bound(),
            durations_per_mode={
                t: {
                    m: source_problem.get_task_mode_duration(t, m)
                    for m in source_problem.get_task_modes(t)
                }
                for t in source_problem.tasks_list
            },
            resource_consumptions={
                t: {
                    m: {
                        r: source_problem.get_cumulative_resource_consumption(r, t, m)
                        for r in source_problem.non_skill_cumulative_resources_list
                    }
                    for m in source_problem.get_task_modes(t)
                }
                for t in source_problem.tasks_list
            },
            successors=source_problem.get_precedence_constraints(),
            unary_resources=source_problem.unary_resources_list,
            unary_resources_skills={
                ur: {
                    s: source_problem.get_unary_resource_skill_value(ur, s)
                    for s in source_problem.skills_list
                }
                for ur in source_problem.unary_resources_list
            },
            unary_resources_availabilities={
                team: [
                    (st, end)
                    for st, end, val in source_problem.get_resource_availabilities(team)
                ]
                for team in source_problem.unary_resources_list
            },
            unary_resources_task_compatibility={
                task: {
                    team
                    for team in source_problem.unary_resources_list
                    if source_problem.is_compatible_task_unary_resource(task, team)
                }
                for task in source_problem.tasks_list
            },
            skills=source_problem.skills_list,
            non_skill_cumulative_resources={
                r: source_problem.get_resource_availabilities(r)
                for r in source_problem.non_renewable_resources_list
            },
            non_renewable_resources={
                nr: source_problem.get_non_renewable_resource_capacity(nr)
                for nr in source_problem.non_renewable_resources_list
            },
            time_windows={
                t: (
                    source_problem.get_task_start_or_end_lower_bound(
                        t, StartOrEnd.START
                    ),
                    source_problem.get_task_start_or_end_lower_bound(t, StartOrEnd.END),
                    source_problem.get_task_start_or_end_upper_bound(
                        t, StartOrEnd.START
                    ),
                    source_problem.get_task_start_or_end_upper_bound(t, StartOrEnd.END),
                )
                for t in source_problem.tasks_list
            },
            flexible_gap_blocking_constraints=source_problem.get_flexible_gap_blocking_constraints(),
            span_blocking_constraints=source_problem.get_span_blocking_constraints(),
            mode_constraints=source_problem.get_mode_constraints(),
            same_unary_allocation=source_problem.get_same_unary_allocation(),
            mode_costs={
                t: {
                    m: source_problem.get_mode_cost(t, m)
                    for m in source_problem.get_task_modes(t)
                }
                for t in source_problem.tasks_list
            },
            unary_resource_costs={
                t: {
                    m: {
                        ur: source_problem.get_unary_resource_cost(t, m, ur)
                        for ur in source_problem.unary_resources_list
                    }
                    for m in source_problem.get_task_modes(t)
                }
                for t in source_problem.tasks_list
            },
            objective=self.objective,
            custom_evaluate_fn=self.custom_evaluate_fn,
            objective_resource_weights=self.objective_resource_weights,
        )
