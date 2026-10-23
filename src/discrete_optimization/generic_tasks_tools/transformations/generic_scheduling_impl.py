#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from abc import abstractmethod
from typing import Generic, Optional, TypeVar

from discrete_optimization.generic_tasks_tools.enums import MinOrMax, StartOrEnd
from discrete_optimization.generic_tasks_tools.generic_scheduling import (
    GenericSchedulingProblem,
    GenericSchedulingSolution,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    AvailabilityIntervals,
    BatchingData,
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
    RawSolution,
    TaskVariable,
)
from discrete_optimization.generic_tasks_tools.objectives.objective_computer import (
    ObjectiveComputer,
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

    def __init__(self, list_objective_computer: list[ObjectiveComputer] = None):
        self._list_objective_computer = list_objective_computer

    @abstractmethod
    def transform_solution_from_raw_generic_to_specific(
        self,
        raw_sol: RawSolution[Task, UnaryResource, Skill],
        source_problem: SpecificSchedulingProblem,
    ) -> SpecificSchedulingSolution:
        """Convert a raw solution (from generic problem) into a specific solution to the source problem.

        Args:
            raw_sol: Raw solution to convert to a specific solution
            source_problem: Target Problem we want to build a solution for.

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

    def get_list_objective_computer(
        self,
        source_problem: SpecificSchedulingProblem,
    ) -> list[ObjectiveComputer]:
        """From the source problem, compute a list of objective computers to be used in `GenericSchedulingImplProblem`.
        To be overriden in child classes.

        Returns:

        """
        # We put None as problem, it will be linked to its parent scheduling problem, when calling __init__
        # of GenericSchedulingImplProblem
        if self._list_objective_computer is not None:
            return self._list_objective_computer
        return source_problem.get_list_objective_computer()

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
        (
            resources_consumption,
            resources_consumption_dependent,
        ) = _extract_resource_consumptions(source_problem)
        successors: dict[Task, set[Task]] = {
            k: set(v) for k, v in source_problem.get_precedence_constraints().items()
        }
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
        optional_tasks = set(source_problem.optional_tasks_list)
        list_objective_computer = self.get_list_objective_computer(source_problem)
        exclusion_resource_capacity = {
            r: source_problem.get_capacity_exclusion_resource(r)
            for r in source_problem.exclusion_resources_list
        }
        exclusion_resource_consumptions = {
            t: {
                m: {
                    r: source_problem.get_task_consumption_exclusion_resource(
                        resource=r, task=t, mode=m
                    )
                    for r in source_problem.exclusion_resources_list
                }
                for m in source_problem.get_task_modes(t)
            }
            for t in source_problem.tasks_list
        }
        exclusion_resource_boolean = {
            t: {
                m: {
                    r: source_problem.is_task_mode_excluding_others(
                        task=t, mode=m, resource=r
                    )
                    for r in source_problem.exclusion_resources_list
                }
                for m in source_problem.get_task_modes(t)
            }
            for t in source_problem.tasks_list
        }

        # Extract batching data
        batching_data = _extract_batching_data(source_problem)

        # Extract entity constraints
        entity_precedence_constraints = (
            source_problem.get_entity_precedence_constraints()
        )
        entity_start_to_start_min_time_lags = source_problem.get_entity_time_lags(
            StartOrEnd.START, StartOrEnd.START, MinOrMax.MIN
        )
        entity_start_to_start_max_time_lags = source_problem.get_entity_time_lags(
            StartOrEnd.START, StartOrEnd.START, MinOrMax.MAX
        )
        entity_start_to_end_min_time_lags = source_problem.get_entity_time_lags(
            StartOrEnd.START, StartOrEnd.END, MinOrMax.MIN
        )
        entity_start_to_end_max_time_lags = source_problem.get_entity_time_lags(
            StartOrEnd.START, StartOrEnd.END, MinOrMax.MAX
        )
        entity_end_to_start_min_time_lags = source_problem.get_entity_time_lags(
            StartOrEnd.END, StartOrEnd.START, MinOrMax.MIN
        )
        entity_end_to_start_max_time_lags = source_problem.get_entity_time_lags(
            StartOrEnd.END, StartOrEnd.START, MinOrMax.MAX
        )
        entity_end_to_end_min_time_lags = source_problem.get_entity_time_lags(
            StartOrEnd.END, StartOrEnd.END, MinOrMax.MIN
        )
        entity_end_to_end_max_time_lags = source_problem.get_entity_time_lags(
            StartOrEnd.END, StartOrEnd.END, MinOrMax.MAX
        )

        # Extract entity time window bounds
        entity_time_window_bounds = _extract_entity_time_window_bounds(source_problem)

        # Get batch entities
        batch_entities = source_problem.get_batch_entities()

        return GenericSchedulingImplProblem(
            horizon=horizon,
            durations_per_mode=durations_per_mode,
            resource_consumptions=resources_consumption,
            resource_consumptions_dependent=resources_consumption_dependent,
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
            flexible_gap_blocking_constraints=source_problem.get_flexible_gap_blocking_constraints(),
            span_blocking_constraints=source_problem.get_span_blocking_constraints(),
            mode_constraints=source_problem.get_mode_constraints(),
            same_unary_allocation=source_problem.get_same_unary_allocation(),
            optional_tasks=optional_tasks,
            list_objective_computer=list_objective_computer,
            alternative_scheduling_subproblems=source_problem.get_alternative_scheduling_subproblem(),
            calendar_preemptive_tasks=source_problem.get_all_tasks_calendar_preempted(),
            exclusion_resource_capacity=exclusion_resource_capacity,
            exclusion_resource_consumptions=exclusion_resource_consumptions,
            exclusion_resource_boolean=exclusion_resource_boolean,
            # Batching data
            batch_entities=batch_entities,
            batching_data=batching_data,
            # Entity constraints
            entity_precedence_constraints=entity_precedence_constraints,
            entity_start_to_start_min_time_lags=entity_start_to_start_min_time_lags,
            entity_start_to_start_max_time_lags=entity_start_to_start_max_time_lags,
            entity_start_to_end_min_time_lags=entity_start_to_end_min_time_lags,
            entity_start_to_end_max_time_lags=entity_start_to_end_max_time_lags,
            entity_end_to_start_min_time_lags=entity_end_to_start_min_time_lags,
            entity_end_to_start_max_time_lags=entity_end_to_start_max_time_lags,
            entity_end_to_end_min_time_lags=entity_end_to_end_min_time_lags,
            entity_end_to_end_max_time_lags=entity_end_to_end_max_time_lags,
            entity_time_window_bounds=entity_time_window_bounds,
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


def _extract_entity_time_window_bounds(
    source_problem: GenericSchedulingProblem,
) -> dict:
    """Extract entity time window bounds from a scheduling problem.

    Tries to get time window bounds for all entities mentioned in entity constraints.

    Returns:
        Dictionary mapping (entity, StartOrEnd) to (lower_bound, upper_bound)
    """

    # Collect all entities from entity constraints
    entities = set()

    # From precedence constraints
    for (
        entity1,
        successors,
    ) in source_problem.get_entity_precedence_constraints().items():
        entities.add(entity1)
        entities.update(successors)

    # From time lag constraints (all 8 types)
    for entity1_se, entity2_se, min_max in [
        (StartOrEnd.START, StartOrEnd.START, MinOrMax.MIN),
        (StartOrEnd.START, StartOrEnd.START, MinOrMax.MAX),
        (StartOrEnd.START, StartOrEnd.END, MinOrMax.MIN),
        (StartOrEnd.START, StartOrEnd.END, MinOrMax.MAX),
        (StartOrEnd.END, StartOrEnd.START, MinOrMax.MIN),
        (StartOrEnd.END, StartOrEnd.START, MinOrMax.MAX),
        (StartOrEnd.END, StartOrEnd.END, MinOrMax.MIN),
        (StartOrEnd.END, StartOrEnd.END, MinOrMax.MAX),
    ]:
        for entity1, entity2, _ in source_problem.get_entity_time_lags(
            entity1_se, entity2_se, min_max
        ):
            entities.add(entity1)
            entities.add(entity2)

    # Get batch entities
    entities.update(source_problem.get_batch_entities())

    # Extract time window bounds for each entity
    entity_time_window_bounds = {}
    for entity in entities:
        for start_or_end in [StartOrEnd.START, StartOrEnd.END]:
            lower_bound = source_problem.get_entity_time_window_lower_bound(
                entity, start_or_end
            )
            upper_bound = source_problem.get_entity_time_window_upper_bound(
                entity, start_or_end
            )
            if lower_bound is not None or upper_bound is not None:
                entity_time_window_bounds[(entity, start_or_end)] = (
                    lower_bound,
                    upper_bound,
                )

    return entity_time_window_bounds


def _extract_batching_data(
    source_problem: GenericSchedulingProblem,
) -> BatchingData:
    """Extract batching data from a scheduling problem.

    Queries all batching-related methods and builds a BatchingData object.

    Returns:
        BatchingData containing all batching parameters
    """

    # Get batch IDs
    batch_ids = source_problem.get_batch_ids()

    # Fixed batches
    fixed_batches = source_problem.get_fixed_batches()

    # Batch size bounds (min/max number of tasks)
    batch_size_bounds = {
        batch_id: source_problem.get_batch_size_bounds(batch_id)
        for batch_id in batch_ids
    }

    # Batch optional flags
    batch_optional = {
        batch_id: source_problem.is_batch_optional(batch_id) for batch_id in batch_ids
    }

    # Task sizes
    task_sizes = {
        task: source_problem.get_task_size(task)
        for task in source_problem.tasks_list
        if source_problem.get_task_size(task) != 1  # Only store non-default
    }

    # Task attributes
    task_attributes = {
        task: attr
        for task in source_problem.tasks_list
        if (attr := source_problem.get_task_attribute(task)) is not None
    }

    # Task batching requirements
    task_batching_mandatory = {
        task
        for task in source_problem.tasks_list
        if source_problem.must_be_batched(task)
    }
    task_batching_forbidden = {
        task
        for task in source_problem.tasks_list
        if not source_problem.can_be_batched(task)
    }

    # Batch capacities
    batch_capacities = {
        batch_id: source_problem.get_batch_capacity(batch_id)
        for batch_id in batch_ids
        if source_problem.get_batch_capacity(batch_id)
        != float("inf")  # Only store non-default
    }

    # Batch duration bounds (if any have been set)
    # This is complex as it depends on task assignments, so we skip it in transformation
    batch_duration_bounds = {}

    # Task compatibility (find incompatible pairs)
    incompatible_task_pairs = set()
    tasks = source_problem.tasks_list
    for i, task1 in enumerate(tasks):
        for task2 in tasks[i + 1 :]:
            if not source_problem.can_batch_together(task1, task2):
                incompatible_task_pairs.add(frozenset([task1, task2]))
    # Batch presence chains
    batch_presence_chains = source_problem.get_batch_presence_chains()

    return BatchingData(
        fixed_batches=fixed_batches,
        batch_size_bounds=batch_size_bounds,
        batch_optional=batch_optional,
        task_sizes=task_sizes,
        task_attributes=task_attributes,
        task_batching_mandatory=task_batching_mandatory,
        task_batching_forbidden=task_batching_forbidden,
        batch_capacities=batch_capacities,
        batch_duration_bounds=batch_duration_bounds,
        incompatible_task_pairs=incompatible_task_pairs,
        batch_presence_chains=batch_presence_chains,
    )


def _extract_resource_consumptions(
    source_problem: GenericSchedulingProblem,
) -> tuple[
    dict[Task, dict[int, dict[CumulativeResource | NonRenewableResource, int]]],
    dict[
        Task,
        dict[
            int,
            dict[
                CumulativeResource | NonRenewableResource,
                dict[frozenset[tuple[Task, int]], int],
            ],
        ],
    ],
]:
    """Extract static and dependent resource consumptions from a scheduling problem.

    Returns:
        Tuple of (static_consumptions, dependent_consumptions)
    """
    resources_consumption: dict[
        Task, dict[int, dict[CumulativeResource | NonRenewableResource, int]]
    ] = {}
    resources_consumption_dependent: dict[
        Task,
        dict[
            int,
            dict[
                CumulativeResource | NonRenewableResource,
                dict[frozenset[tuple[Task, int]], int],
            ],
        ],
    ] = {}

    for task in source_problem.tasks_list:
        resources_consumption[task] = {}
        resources_consumption_dependent[task] = {}
        for mode in source_problem.get_task_modes(task=task):
            resources_consumption[task][mode] = {}
            resources_consumption_dependent[task][mode] = {}

            # Non-renewable resources
            for resource in source_problem.non_renewable_resources_list:
                if source_problem.is_non_renewable_resource_task_mode_consumption_dependent(
                    resource=resource, task=task, mode=mode
                ):
                    resources_consumption_dependent[task][mode][resource] = (
                        source_problem.get_non_renewable_resource_consumption_mapping(
                            resource=resource, task=task, mode=mode
                        )
                    )
                else:
                    conso = source_problem.get_non_renewable_resource_consumption(
                        resource=resource, task=task, mode=mode
                    )
                    if conso != 0:
                        resources_consumption[task][mode][resource] = conso

            # Cumulative resources
            for resource in source_problem.cumulative_resources_list:
                if source_problem.is_cumulative_resource_task_mode_consumption_dependent(
                    resource=resource, task=task, mode=mode
                ):
                    resources_consumption_dependent[task][mode][resource] = (
                        source_problem.get_cumulative_resource_consumption_mapping(
                            resource=resource, task=task, mode=mode
                        )
                    )
                else:
                    conso = source_problem.get_cumulative_resource_consumption(
                        resource=resource, task=task, mode=mode
                    )
                    if conso != 0:
                        resources_consumption[task][mode][resource] = conso

    return resources_consumption, resources_consumption_dependent


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
                    is_present=solution.is_present(task),
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
