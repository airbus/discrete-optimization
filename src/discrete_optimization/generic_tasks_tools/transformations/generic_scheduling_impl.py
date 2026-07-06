#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from abc import abstractmethod
from collections.abc import Callable, Iterable
from typing import Generic, Optional, TypeVar

from discrete_optimization.generic_tasks_tools.enums import MinOrMax, StartOrEnd
from discrete_optimization.generic_tasks_tools.generic_scheduling import (
    GenericSchedulingProblem,
    GenericSchedulingSolution,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    AnyResource,
    AvailabilityIntervals,
    CumulativeResource,
    GenericSchedulingImplProblem,
    GenericSchedulingImplSolution,
    NonRenewableResource,
    NonSkillCumulativeResource,
    Skill,
    Task,
    UnaryAvailabilityIntervals,
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import (
    Objective,
    RawSolution,
    TaskVariable,
)
from discrete_optimization.generic_tools.transformation import ProblemTransformation

SpecificSchedulingProblem = TypeVar(
    "SpecificSchedulingProblem", bound=GenericSchedulingProblem
)
SpecificSchedulingSolution = TypeVar(
    "SpecificSchedulingSolution", bound=GenericSchedulingSolution
)


class ToGenericSchedulingImpl(
    ProblemTransformation[
        SpecificSchedulingProblem,
        SpecificSchedulingSolution,
        GenericSchedulingImplProblem,
        GenericSchedulingImplSolution,
    ],
    Generic[
        SpecificSchedulingProblem,
        SpecificSchedulingSolution,
    ],
):
    """Transform a specific scheduling problem into the generic implementation.

    This is still an abstract class, as `convert_solution_from_raw_generic_to_specific()` remains to implement.


    """

    @abstractmethod
    def transform_solution_from_raw_generic_to_specific(
        self,
        raw_sol: RawSolution[Task, UnaryResource, Skill],
        source_problem: SpecificSchedulingProblem,
    ) -> SpecificSchedulingSolution:
        """Convert a raw solution (from generic problem) into a specific solution to the source problem.

        Args:
            source_problem:

        Returns:

        """
        ...

    def back_transform_solution(
        self,
        solution: GenericSchedulingImplSolution,
        source_problem: SpecificSchedulingProblem,
    ) -> SpecificSchedulingSolution:
        return self.transform_solution_from_raw_generic_to_specific(
            raw_sol=solution.raw_sol, source_problem=source_problem
        )

    def transform_objective(
        self,
        source_problem: SpecificSchedulingProblem,
    ) -> tuple[
        Objective | Iterable[tuple[Objective, int]],
        Optional[Callable[[GenericSchedulingImplSolution], int]],
        Optional[dict[AnyResource, int]],
        bool,
    ]:
        """Transform scheduling problem objective for the generic implementation.

        This default implementation returns default values of `GenericSchedulingImplProblem`.
        To be overriden in child classes.

        Returns:
            objective, custom_evaluate_fn, objective_resource_weights, compute_time_penalty: the corresponding arguments
            `of GenericSchedulingImplProblem.__init__()`, resulting to computing makespan + time penalty.
            See its documentation for more details.


        """
        objective = Objective.MAKESPAN
        custom_evaluate_fn = None  # No custom objective
        objective_resource_weights = None  # All resources have same weights
        compute_time_penalty = True  # Time penalty will be computed during `evaluate()`
        return (
            objective,
            custom_evaluate_fn,
            objective_resource_weights,
            compute_time_penalty,
        )

    def transform_problem(
        self,
        source_problem: SpecificSchedulingProblem,
    ) -> GenericSchedulingImplProblem:
        horizon = source_problem.get_makespan_upper_bound()
        durations_per_mode: dict[Task, dict[int, int]] = {
            task: {
                mode: source_problem.get_task_mode_duration(task=task, mode=mode)
                for mode in source_problem.get_task_modes(task=task)
            }
            for task in source_problem.tasks_list
        }
        resources_consumption: dict[
            Task, dict[int, dict[CumulativeResource | NonRenewableResource, int]]
        ] = {
            task: {
                mode: {
                    resource: source_problem.get_non_renewable_resource_consumption(
                        resource=resource, task=task, mode=mode
                    )
                    for resource in source_problem.non_renewable_resources_list
                }
                | {
                    resource: source_problem.get_cumulative_resource_consumption(
                        resource=resource, task=task, mode=mode
                    )
                    for resource in source_problem.cumulative_resources_list
                }
                for mode in source_problem.get_task_modes(task=task)
            }
            for task in source_problem.tasks_list
        }
        successors: dict[Task, Iterable[Task]] = (
            source_problem.get_precedence_constraints()
        )
        unary_resources: set[UnaryResource] = set(source_problem.unary_resources_list)
        unary_resources_skills: dict[UnaryResource, dict[Skill, int]] = {
            unary_resource: {
                skill: value
                for skill in source_problem.skills_list
                if (
                    value := source_problem.get_unary_resource_skill_value(
                        unary_resource=unary_resource, skill=skill
                    )
                )
                > 0
            }
            for unary_resource in source_problem.unary_resources_list
        }
        unary_resources_availabilities: dict[
            UnaryResource, UnaryAvailabilityIntervals
        ] = {
            unary_resource: [
                (start, end)
                for start, end, value in source_problem.get_resource_availabilities(
                    resource=unary_resource
                )
                if value > 0
            ]
            for unary_resource in source_problem.unary_resources_list
        }
        unary_resources_task_compatibility: dict[Task, set[UnaryResource]] = {
            task: {
                unary_resource
                for unary_resource in source_problem.unary_resources_list
                if source_problem.is_compatible_task_unary_resource(
                    task=task, unary_resource=unary_resource
                )
            }
            for task in source_problem.tasks_list
        }
        skills: set[Skill] = set(source_problem.skills_list)
        non_skill_cumulative_resources: dict[
            NonSkillCumulativeResource, int | AvailabilityIntervals
        ] = {
            resource: source_problem.get_resource_availabilities(resource=resource)
            for resource in source_problem.non_skill_cumulative_resources_list
        }
        non_renewable_resources: dict[NonRenewableResource, int] = {
            resource: source_problem.get_non_renewable_resource_capacity(
                resource=resource
            )
            for resource in source_problem.non_renewable_resources_list
        }
        time_windows: dict[
            Task, tuple[int | None, int | None, int | None, int | None]
        ] = {
            task: (
                source_problem.get_task_start_or_end_lower_bound(
                    task=task, start_or_end=StartOrEnd.START
                ),
                source_problem.get_task_start_or_end_lower_bound(
                    task=task, start_or_end=StartOrEnd.END
                ),
                source_problem.get_task_start_or_end_upper_bound(
                    task=task, start_or_end=StartOrEnd.START
                ),
                source_problem.get_task_start_or_end_upper_bound(
                    task=task, start_or_end=StartOrEnd.END
                ),
            )
            for task in source_problem.tasks_list
        }
        start_to_start_min_time_lags: list[tuple[Task, Task, int]] = (
            _construct_min_only_time_lags_from_min_and_max_time_lags(
                source_problem=source_problem,
                task1_start_or_end=StartOrEnd.START,
                task2_start_or_end=StartOrEnd.START,
            )
        )
        start_to_end_min_time_lags: list[tuple[Task, Task, int]] = (
            _construct_min_only_time_lags_from_min_and_max_time_lags(
                source_problem=source_problem,
                task1_start_or_end=StartOrEnd.START,
                task2_start_or_end=StartOrEnd.END,
            )
        )
        end_to_start_min_time_lags: list[tuple[Task, Task, int]] = (
            _construct_min_only_time_lags_from_min_and_max_time_lags(
                source_problem=source_problem,
                task1_start_or_end=StartOrEnd.END,
                task2_start_or_end=StartOrEnd.START,
            )
        )
        end_to_end_min_time_lags: list[tuple[Task, Task, int]] = (
            _construct_min_only_time_lags_from_min_and_max_time_lags(
                source_problem=source_problem,
                task1_start_or_end=StartOrEnd.END,
                task2_start_or_end=StartOrEnd.END,
            )
        )
        no_overlap_sets: set[frozenset[Task]] = source_problem.get_no_overlap()
        forbidden_intervals: dict[Task, list[tuple[int, int]]] = {
            task: source_problem.get_forbidden_intervals(task)
            for task in source_problem.tasks_list
        }
        mode_costs: dict[Task, dict[int, int]] = {
            task: {
                mode: source_problem.get_mode_cost(task=task, mode=mode)
                for mode in source_problem.get_task_modes(task=task)
            }
            for task in source_problem.tasks_list
        }
        unary_resource_costs: dict[Task, dict[int, dict[UnaryResource, int]]] = {
            task: {
                mode: {
                    unary_resource: source_problem.get_unary_resource_cost(
                        unary_resource=unary_resource, task=task, mode=mode
                    )
                    for unary_resource in source_problem.unary_resources_list
                }
                for mode in source_problem.get_task_modes(task=task)
            }
            for task in source_problem.tasks_list
        }
        (
            objective,
            custom_evaluate_fn,
            objective_resource_weights,
            compute_time_penalty,
        ) = self.transform_objective(source_problem)

        return GenericSchedulingImplProblem(
            horizon=horizon,
            durations_per_mode=durations_per_mode,
            resource_consumptions=resources_consumption,
            successors=successors,
            unary_resources=unary_resources,
            unary_resources_skills=unary_resources_skills,
            unary_resources_availabilities=unary_resources_availabilities,
            unary_resources_task_compatibility=unary_resources_task_compatibility,
            skills=skills,
            non_skill_cumulative_resources=non_skill_cumulative_resources,
            non_renewable_resources=non_renewable_resources,
            time_windows=time_windows,
            start_to_start_min_time_lags=start_to_start_min_time_lags,
            start_to_end_min_time_lags=start_to_end_min_time_lags,
            end_to_start_min_time_lags=end_to_start_min_time_lags,
            end_to_end_min_time_lags=end_to_end_min_time_lags,
            no_overlap_sets=no_overlap_sets,
            forbidden_intervals=forbidden_intervals,
            mode_costs=mode_costs,
            unary_resource_costs=unary_resource_costs,
            objective=objective,
            custom_evaluate_fn=custom_evaluate_fn,
            objective_resource_weights=objective_resource_weights,
            compute_time_penalty=compute_time_penalty,
        )

    def forward_transform_solution(
        self,
        solution: SpecificSchedulingSolution,
        target_problem: GenericSchedulingImplProblem,
    ) -> Optional[GenericSchedulingImplSolution]:
        return convert_solution_from_specific_to_generic(
            solution=solution,
            generic_problem=target_problem,
        )

    def is_bidirectional(self, source_problem: SpecificSchedulingProblem) -> bool:
        return True


def _construct_min_only_time_lags_from_min_and_max_time_lags(
    source_problem: GenericSchedulingProblem,
    task1_start_or_end: StartOrEnd,
    task2_start_or_end: StartOrEnd,
) -> list[tuple[Task, Task, int]]:
    return source_problem.get_original_time_lags(
        task1_start_or_end=task1_start_or_end,
        task2_start_or_end=task2_start_or_end,
        min_or_max=MinOrMax.MIN,
    ) + [
        (task2, task1, -offset)
        for task1, task2, offset in source_problem.get_original_time_lags(
            task1_start_or_end=task2_start_or_end,
            task2_start_or_end=task1_start_or_end,
            min_or_max=MinOrMax.MAX,
        )
    ]


def convert_solution_from_specific_to_generic(
    solution: SpecificSchedulingSolution,
    generic_problem: GenericSchedulingImplProblem,
) -> GenericSchedulingImplSolution:
    return GenericSchedulingImplSolution(
        problem=generic_problem,
        raw_sol=RawSolution(
            task_variables={
                task: TaskVariable(
                    start=solution.get_start_time(task),
                    end=solution.get_end_time(task),
                    mode=solution.get_mode(task),
                    allocated={
                        unary_resource: {
                            skill
                            for skill in generic_problem.skills_list
                            if solution.is_skill_used(
                                task=task,
                                skill=skill,
                                unary_resource=unary_resource,
                            )
                        }
                        for unary_resource in solution.get_task_allocation(task)
                    },
                )
                for task in generic_problem.tasks_list
            }
        ),
    )


class FromGenericSchedulingImpl(
    ProblemTransformation[
        GenericSchedulingImplProblem,
        GenericSchedulingImplSolution,
        SpecificSchedulingProblem,
        SpecificSchedulingSolution,
    ],
    Generic[SpecificSchedulingProblem, SpecificSchedulingSolution],
):
    """Transform the generic implementation of a scheduling problem into a specific one.

    This is still an abstract class, as the `transform_problem()` remains to implement.

    """

    def back_transform_solution(
        self,
        solution: SpecificSchedulingSolution,
        source_problem: GenericSchedulingImplProblem,
    ) -> GenericSchedulingImplSolution:
        return convert_solution_from_specific_to_generic(
            solution=solution,
            generic_problem=source_problem,
        )
