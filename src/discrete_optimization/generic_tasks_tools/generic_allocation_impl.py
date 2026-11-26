#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field

from discrete_optimization.generic_tasks_tools.alternative_subproblems import (
    AlternativeSchedulingSubProblem,
)
from discrete_optimization.generic_tasks_tools.entities import SchedulingEntity
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.generic_allocation import (
    GenericAllocationProblem,
    GenericAllocationSolution,
    NoNonSkillCumulativeResource,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_impl import (
    AvailabilityIntervals,
    BatchingData,
    ExclusionResource,
    GenericSchedulingImplProblem,
    GenericSchedulingImplSolution,
    NonRenewableResource,
    Skill,
    Task,
    UnaryAvailabilityIntervals,
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import (
    RawSolution,
)
from discrete_optimization.generic_tasks_tools.multimode import ModeConstraintType
from discrete_optimization.generic_tasks_tools.objectives.objective_computer import (
    ObjectiveComputer,
)
from discrete_optimization.generic_tasks_tools.objectives.unary_resource_used import (
    UnaryResourcesUsedComputer,
)
from discrete_optimization.generic_tasks_tools.resource_blocking import (
    FlexibleGapBlockingConstraint,
    SpanBlockingConstraint,
)
from discrete_optimization.generic_tasks_tools.routing import (
    OrderedRoutingEntity,
    RoutingEntity,
)

CumulativeResource = Skill


@dataclass(kw_only=True)
class GenericAllocationImplProblem(
    GenericSchedulingImplProblem,
    GenericAllocationProblem[
        Task,
        UnaryResource,
        Skill,
        NonRenewableResource,
        ExclusionResource,
    ],
):
    """Generic implementation of a allocation problem.

    It implements the abstract class `GenericAllocationProblem`.

    Attributes:
        tasks_modes: task -> available modes. Will be used to know available tasks and modes.
        resource_consumptions: task -> mode -> resource -> conso.
            Skill or non-renewable resource consumption, task by task, mode by mode.
            Missing key => conso = 0
        unary_resources: available unary resources.  Default to none.
        unary_resources_skills: skill values of each unary resource. Missing key => skill value = 0
        unary_resources_availabilities: availability of unary resources on the form of list of intervals (start, end)
            Missing key => always available.
        unary_resources_task_compatibility: maps a task to its compatible unary resources. Missing key => all unary resources allowed.
        skills: available skills
        non_renewable_resources: non-renewable resources max capacities
        same_unary_allocation: list of tasks subsets that must have same allocation
        incompatible_unary_allocation: list of tasks subsets that can not share same allocated unary resource

    """

    tasks_modes: dict[Task, set[int]]
    resource_consumptions: dict[
        Task, dict[int, dict[CumulativeResource | NonRenewableResource, int]]
    ] = field(default_factory=dict)
    resource_consumptions_dependent: dict[
        Task,
        dict[
            int,
            dict[
                CumulativeResource | NonRenewableResource,
                dict[frozenset[tuple[Task, int]], int],
            ],
        ],
    ] = field(default_factory=dict)
    unary_resource_consumptions_dependent: dict[
        Task,
        dict[
            int,
            dict[CumulativeResource | NonRenewableResource, dict[UnaryResource, int]],
        ],
    ] = field(default_factory=dict)
    unary_resources: set[UnaryResource] = field(default_factory=set)
    unary_resources_skills: dict[UnaryResource, dict[Skill, int]] = field(
        default_factory=dict
    )
    unary_resources_task_compatibility: dict[Task, set[UnaryResource]] = field(
        default_factory=dict
    )
    unary_resources_task_mode_compatibility: dict[
        Task, dict[UnaryResource, set[int]]
    ] = field(default_factory=dict)
    skills: set[Skill] = field(default_factory=set)
    non_renewable_resources: dict[NonRenewableResource, int] = field(
        default_factory=dict
    )
    mode_constraints: list[tuple[ModeConstraintType, list[tuple[Task, int]]]] = field(
        default_factory=list
    )
    mode_subset_constraints: list[
        tuple[ModeConstraintType, list[tuple[Task, set[int]]]]
    ] = field(default_factory=list)
    same_unary_allocation: list[set[Task]] = field(default_factory=list)
    incompatible_unary_allocation: list[set[Task]] = field(default_factory=list)
    optional_tasks: set[Task] = field(default_factory=set)
    list_objective_computer: list[ObjectiveComputer] = field(
        compare=False, default=None
    )
    alternative_scheduling_subproblems: list[AlternativeSchedulingSubProblem] = field(
        default_factory=list
    )
    exclusion_resource_capacity: dict[ExclusionResource, int] = field(
        default_factory=dict
    )
    exclusion_resource_consumptions: dict[
        Task, dict[int, dict[ExclusionResource, int]]
    ] = field(default_factory=dict)
    exclusion_resource_boolean: dict[Task, dict[int, dict[ExclusionResource, bool]]] = (
        field(default_factory=dict)
    )
    # Specify custom constraint on presence of tasks
    linear_cstrs_presence_tasks: list[
        tuple[list[tuple[Task, int]], int | None, int | None]
    ] = field(default_factory=list)

    # Fixed attributes from GenericSchedulingImplProblem
    horizon: int = field(default=1, init=False)
    durations_per_mode: dict[Task, dict[int, int]] = field(init=False)
    successors: dict[Task, set[Task]] = field(init=False, default_factory=dict)
    non_skill_cumulative_resources: dict[
        NoNonSkillCumulativeResource, int | AvailabilityIntervals
    ] = field(init=False, default_factory=dict)
    unary_resources_availabilities: dict[UnaryResource, UnaryAvailabilityIntervals] = (
        field(init=False, default_factory=dict)
    )
    time_windows: dict[Task, tuple[int | None, int | None, int | None, int | None]] = (
        field(init=False, default_factory=dict)
    )
    start_to_start_min_time_lags: list[tuple[Task, Task, int]] = field(
        init=False, default_factory=list
    )
    start_to_end_min_time_lags: list[tuple[Task, Task, int]] = field(
        init=False, default_factory=list
    )
    end_to_start_min_time_lags: list[tuple[Task, Task, int]] = field(
        init=False, default_factory=list
    )
    end_to_end_min_time_lags: list[tuple[Task, Task, int]] = field(
        init=False, default_factory=list
    )
    no_overlap_sets: set[frozenset[Task]] = field(init=False, default_factory=set)
    forbidden_intervals: dict[Task, list[tuple[int, int]]] = field(
        init=False, default_factory=dict
    )
    flexible_gap_blocking_constraints: list[FlexibleGapBlockingConstraint] = field(
        init=False, default_factory=list
    )
    span_blocking_constraints: list[SpanBlockingConstraint] = field(
        init=False, default_factory=list
    )
    calendar_preemptive_tasks: set[Task] = field(init=False, default_factory=set)
    entity_precedence_constraints: dict[
        SchedulingEntity, Iterable[SchedulingEntity]
    ] = field(init=False, default_factory=dict)
    entity_start_to_start_min_time_lags: list[
        tuple[SchedulingEntity, SchedulingEntity, int]
    ] = field(init=False, default_factory=list)
    entity_start_to_start_max_time_lags: list[
        tuple[SchedulingEntity, SchedulingEntity, int]
    ] = field(init=False, default_factory=list)
    entity_end_to_start_min_time_lags: list[
        tuple[SchedulingEntity, SchedulingEntity, int]
    ] = field(init=False, default_factory=list)
    entity_end_to_start_max_time_lags: list[
        tuple[SchedulingEntity, SchedulingEntity, int]
    ] = field(init=False, default_factory=list)
    entity_end_to_end_min_time_lags: list[
        tuple[SchedulingEntity, SchedulingEntity, int]
    ] = field(init=False, default_factory=list)
    entity_end_to_end_max_time_lags: list[
        tuple[SchedulingEntity, SchedulingEntity, int]
    ] = field(init=False, default_factory=list)
    entity_start_to_end_min_time_lags: list[
        tuple[SchedulingEntity, SchedulingEntity, int]
    ] = field(init=False, default_factory=list)
    entity_start_to_end_max_time_lags: list[
        tuple[SchedulingEntity, SchedulingEntity, int]
    ] = field(init=False, default_factory=list)
    entity_time_window_bounds: dict[
        tuple[SchedulingEntity, StartOrEnd], tuple[int | None, int | None]
    ] = field(init=False, default_factory=dict)
    batch_entities: list = field(
        init=False, default_factory=list
    )  # list[VariableBatchEntity]
    task_to_batch_assignment: dict[Task, Hashable] = field(
        init=False, default_factory=dict
    )
    batching_data: BatchingData = field(init=False, default_factory=BatchingData)
    list_routing_entities: list[RoutingEntity] = field(init=False, default_factory=list)
    list_ordered_routing_entities: list[OrderedRoutingEntity] = field(
        init=False, default_factory=list
    )

    def update_problem(self):
        """Method to call when some attributes of the problem are modified."""
        self.durations_per_mode = {
            task: {mode: 1 for mode in modes}
            for task, modes in self.tasks_modes.items()
        }
        super().update_problem()

    @property
    def default_objective_computer_list(self) -> list[ObjectiveComputer[Task]]:
        """Default objective: nb of unary resources used."""
        return [UnaryResourcesUsedComputer()]


class GenericAllocationImplSolution(
    GenericSchedulingImplSolution,
    GenericAllocationSolution[
        Task,
        UnaryResource,
        Skill,
        NonRenewableResource,
        ExclusionResource,
    ],
):
    """Generic implementation of a solution to an allocation problem.

    It implements the abstract class `GenericAllocationSolution`.

    Hypothesis: the wrapped RawSolution has tasks starts and ends set to 0.

    """

    problem: GenericAllocationImplProblem

    def __init__(
        self,
        problem: GenericAllocationImplProblem,
        raw_sol: RawSolution[Task, UnaryResource, Skill],
        check_no_scheduling=False,
    ):
        if check_no_scheduling:
            for task, task_variable in raw_sol.task_variables.items():
                assert task_variable.start == 0 and task_variable.end == 1, (
                    "For generic allocation problem, start and end must both set to 0 and 1."
                )
        super().__init__(problem, raw_sol)
