#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any, Generic

from ortools.sat.python.cp_model import IntervalVar, IntVar

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.entities import (
    GroupEntity,
    SchedulingEntity,
    TaskEntity,
    TaskModeEntity,
)
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.generic_scheduling import (
    CumulativeResource,
    GenericSchedulingProblem,
)
from discrete_optimization.generic_tasks_tools.resource_blocking import (
    BlockingConstraintMetadata,
    BlockingMode,
    CumulativeResource,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.cumulative_resource import (
    CumulativeResource,
    CumulativeResourceSchedulingCpSatSolver,
    OtherCalendarResource,
)


class ResourceBlockingCpSatSolver(
    CumulativeResourceSchedulingCpSatSolver[
        Task, CumulativeResource, OtherCalendarResource
    ],
    Generic[Task, CumulativeResource, OtherCalendarResource],
):
    """CP-SAT mixin for handling resource blocking constraints.

    This mixin adds support for:
    - Flexible gap blocking: resources blocked between two scheduling entities
    - Span blocking: resources blocked during the span of a set of tasks

    The mixin handles overlaps between blocking intervals and:
    - Calendar unavailability periods (fake tasks)
    - Actual task execution intervals
    - Other blocking intervals

    It creates appropriate cumulative constraints per resource, properly accounting
    for blocking intervals without double-counting consumption.
    """

    problem: GenericSchedulingProblem
    _blocking_intervals: dict
    _starts_entity: dict[SchedulingEntity, IntVar]
    _ends_entity: dict[SchedulingEntity, IntVar]
    _durations_entity: dict[SchedulingEntity, IntVar]
    _intervals_entity: dict[SchedulingEntity, IntVar]
    _bounds_entity: dict[SchedulingEntity, IntVar]

    def init_model(self, **kwargs: Any) -> None:
        """Initialize model and reset blocking interval storage."""
        super().init_model(**kwargs)
        self._blocking_intervals: dict[
            CumulativeResource,
            list[tuple[IntervalVar, int, BlockingConstraintMetadata]],
        ] = {}

    def constrain_group_entity_times(self, entity: GroupEntity) -> None:
        """Add constraints for group entity start/end times.

        A group's start is the minimum start of its tasks.
        A group's end is the maximum end of its tasks.

        Args:
            entity: The group entity
        """
        group_start = self._starts_entity[entity]
        group_end = self._ends_entity[entity]
        self.cp_model.AddMinEquality(
            group_start,
            [
                self.get_task_start_or_end_variable(task, StartOrEnd.START)
                for task in entity.tasks
            ],
        )
        self.cp_model.AddMaxEquality(
            group_end,
            [
                self.get_task_start_or_end_variable(task, StartOrEnd.END)
                for task in entity.tasks
            ],
        )

    def _get_tasks_from_entity(self, entity: SchedulingEntity) -> set[Task]:
        """Extract all tasks involved in a scheduling entity.

        Args:
            entity: The scheduling entity

        Returns:
            Set of tasks involved in this entity
        """
        if isinstance(entity, TaskEntity):
            return {entity.task}
        elif isinstance(entity, TaskModeEntity):
            return {entity.task}
        elif isinstance(entity, GroupEntity):
            return set(entity.tasks)
        else:
            return set()

    def get_lb_ub_entity(self, entity: SchedulingEntity) -> tuple[int, int, int, int]:
        """Return lbstart, ubstart, lbend, ubend"""
        if isinstance(entity, (TaskEntity, TaskModeEntity)):
            lbs = self.problem.get_task_start_or_end_lower_bound(
                entity.task, StartOrEnd.START
            )
            ubs = self.problem.get_task_start_or_end_upper_bound(
                entity.task, StartOrEnd.START
            )
            lbe = self.problem.get_task_start_or_end_lower_bound(
                entity.task, StartOrEnd.END
            )
            ube = self.problem.get_task_start_or_end_upper_bound(
                entity.task, StartOrEnd.END
            )
            return lbs, ubs, lbe, ube

        lb_start = [
            self.problem.get_task_start_or_end_lower_bound(
                task=task, start_or_end=StartOrEnd.START
            )
            for task in entity.get_tasks()
        ]
        ub_start = [
            self.problem.get_task_start_or_end_upper_bound(
                task=task, start_or_end=StartOrEnd.START
            )
            for task in entity.get_tasks()
        ]
        lb_end = [
            self.problem.get_task_start_or_end_lower_bound(
                task=task, start_or_end=StartOrEnd.END
            )
            for task in entity.get_tasks()
        ]
        ub_end = [
            self.problem.get_task_start_or_end_upper_bound(
                task=task, start_or_end=StartOrEnd.END
            )
            for task in entity.get_tasks()
        ]
        return min(lb_start), max(ub_start), min(lb_end), max(ub_end)

    def get_lb_ub_size(
        self,
        entity1: SchedulingEntity,
        start_or_end1: StartOrEnd,
        entity2: SchedulingEntity,
        start_or_end2: StartOrEnd,
    ):
        lbs1, ubs1, lbe1, ube1 = self.get_lb_ub_entity(entity1)
        lbs2, ubs2, lbe2, ube2 = self.get_lb_ub_entity(entity2)
        if start_or_end1 == StartOrEnd.START:
            if start_or_end2 == StartOrEnd.START:
                return max(0, lbs2 - ubs1), max(0, ubs2 - lbs1)
            if start_or_end2 == StartOrEnd.END:
                return max(0, lbe2 - ubs1), max(0, ube2 - lbs1)
        if start_or_end1 == StartOrEnd.END:
            if start_or_end2 == StartOrEnd.START:
                return max(0, lbs2 - ube1), max(0, ubs2 - lbe1)
            if start_or_end2 == StartOrEnd.END:
                return max(0, lbe2 - ube1), max(0, ube2 - lbe1)
        return None, None

    def create_entity_intervals(self) -> None:
        self._starts_entity = {}
        self._ends_entity = {}
        self._durations_entity = {}
        self._intervals_entity = {}
        all_entities = []
        for (
            entity_1,
            _,
            entity_2,
            _,
            _,
            _,
        ) in self.problem.get_flexible_gap_blocking_constraints():
            all_entities.append(entity_1)
            all_entities.append(entity_2)
        for entity, _, _ in self.problem.get_span_blocking_constraints():
            all_entities.append(entity)

        for entity in all_entities:
            if entity not in self._starts_entity:
                tasks = self._get_tasks_from_entity(entity)
                if len(tasks) == 1:
                    self._starts_entity[entity] = self.get_task_start_or_end_variable(
                        task=entity.task, start_or_end=StartOrEnd.START
                    )
                    self._ends_entity[entity] = self.get_task_start_or_end_variable(
                        task=entity.task, start_or_end=StartOrEnd.END
                    )
                    self._durations_entity[entity] = (
                        self._ends_entity[entity] - self._starts_entity[entity]
                    )
                    self._intervals_entity[entity] = self.get_task_interval(entity.task)
                else:
                    lb_start = [
                        self.problem.get_task_start_or_end_lower_bound(
                            task=task, start_or_end=StartOrEnd.START
                        )
                        for task in tasks
                    ]
                    ub_start = [
                        self.problem.get_task_start_or_end_upper_bound(
                            task=task, start_or_end=StartOrEnd.START
                        )
                        for task in tasks
                    ]
                    lb_end = [
                        self.problem.get_task_start_or_end_lower_bound(
                            task=task, start_or_end=StartOrEnd.END
                        )
                        for task in tasks
                    ]
                    ub_end = [
                        self.problem.get_task_start_or_end_upper_bound(
                            task=task, start_or_end=StartOrEnd.END
                        )
                        for task in tasks
                    ]
                    min_lb_start = min(lb_start)
                    max_ub_start = max(ub_start)
                    min_lb_end = min(lb_end)
                    max_ub_end = max(ub_end)
                    self._starts_entity[entity] = self.cp_model.NewIntVar(
                        lb=min_lb_start, ub=max_ub_start, name=f"start_{entity.tasks}"
                    )
                    self._ends_entity[entity] = self.cp_model.NewIntVar(
                        lb=min_lb_end, ub=max_ub_end, name=f"end_{entity.tasks}"
                    )
                    self._durations_entity[entity] = self.cp_model.NewIntVar(
                        lb=max(0, min_lb_end - max_ub_start),
                        ub=max(0, max_ub_end - min_lb_start),
                        name=f"duration_{entity.tasks}",
                    )
                    self._intervals_entity[entity] = self.cp_model.NewIntervalVar(
                        start=self._starts_entity[entity],
                        end=self._ends_entity[entity],
                        size=self._durations_entity[entity],
                        name=f"interval_{entity.tasks}",
                    )

    def create_flexible_gap_blocking_intervals(self) -> None:
        """Create interval variables for flexible gap blocking constraints.

        For each flexible gap constraint, creates an interval from the first entity's
        reference point to the second entity's reference point, with the specified
        resource demands.
        """
        for i_constraint, constraint in enumerate(
            self.problem.get_flexible_gap_blocking_constraints()
        ):
            (
                entity1,
                ref1,
                entity2,
                ref2,
                resources,
                metadata,
            ) = constraint
            # Get time variables for the gap boundaries
            gap_start = (
                self._starts_entity[entity1]
                if ref1 == StartOrEnd.START
                else self._ends_entity[entity1]
            )
            gap_end = (
                self._starts_entity[entity2]
                if ref2 == StartOrEnd.START
                else self._ends_entity[entity2]
            )
            lb_size, ub_size = self.get_lb_ub_size(entity1, ref1, entity2, ref2)
            # Create a variable for the gap size
            # The gap may be 0 or positive (we'll enforce positive later)
            gap_size = self.cp_model.NewIntVar(
                lb=lb_size,
                ub=ub_size,
                name=f"gap_size_{i_constraint}",
            )
            # Create a boolean variable for gap validity (positive duration)
            gap_is_present = self.cp_model.NewBoolVar(f"gap_valid_{i_constraint}")
            # Constrain gap_size = gap_end - gap_start when gap is present
            self.cp_model.Add(gap_size == gap_end - gap_start).OnlyEnforceIf(
                gap_is_present
            )
            # Create interval variable for the gap
            gap_interval = self.cp_model.NewOptionalIntervalVar(
                start=gap_start,
                size=gap_size,
                end=gap_end,
                is_present=gap_is_present,
                name=f"blocking_gap_{entity1.entity_id}_{ref1}_to_{entity2.entity_id}_{ref2}",
            )
            if not isinstance(entity1, TaskModeEntity) and not isinstance(
                entity2, TaskModeEntity
            ):
                self.cp_model.Add(gap_is_present == 1)

            # For TaskModeEntity, only block when task is in the specified mode
            if isinstance(entity1, TaskModeEntity):
                mode_present = self.get_task_mode_is_present_variable(
                    task=entity1.task, mode=entity1.mode
                )
                self.cp_model.AddImplication(mode_present, gap_is_present)

            if isinstance(entity2, TaskModeEntity):
                mode_present = self.get_task_mode_is_present_variable(
                    task=entity2.task, mode=entity2.mode
                )
                self.cp_model.AddImplication(mode_present, gap_is_present)
            # Store blocking intervals per resource with metadata and involved tasks
            for resource, demand in resources.items():
                if resource not in self._blocking_intervals:
                    self._blocking_intervals[resource] = []
                # Store interval with its metadata and tasks for later processing
                self._blocking_intervals[resource].append(
                    (gap_interval, demand, metadata)
                )

    def create_span_blocking_intervals(self) -> None:
        """Create interval variables for span blocking constraints.

        For each span constraint, creates an interval from the minimum start time
        to the maximum end time of the specified tasks, with the specified resource demands.
        """
        for i_constraint, constraint in enumerate(
            self.problem.get_span_blocking_constraints()
        ):
            entity, resources, metadata = constraint
            # Store blocking intervals per resource with metadata
            for resource, demand in resources.items():
                if resource not in self._blocking_intervals:
                    self._blocking_intervals[resource] = []
                # Store interval with its metadata and task set for overlap handling
                self._blocking_intervals[resource].append(
                    (self._intervals_entity[entity], demand, metadata)
                )

    def create_cumulative_constraint_including_blocking(
        self, resource: CumulativeResource
    ) -> None:
        """Create cumulative constraints including blocking intervals.

        Blocking is ALWAYS ADDITIVE: task consumption + blocking consumption.

        Creates TWO cumulative constraints to properly handle BlockingMode:

        Constraint 1 (WITHOUT calendar):
            - All task intervals
            - ALL blocking intervals (RESERVATION + ACTIVE)
            - NO fake tasks (calendar gaps)
            Purpose: Enforces RESERVATION blocking even during unavailable periods

        Constraint 2 (WITH calendar):
            - All task intervals
            - ONLY ACTIVE blocking intervals
            - Fake tasks (calendar gaps)
            Purpose: Enforces ACTIVE blocking only during available periods

        Args:
            resource: The cumulative resource to constrain
        """
        # Get task consumption intervals
        task_intervals = self.get_resource_consumption_intervals(resource)

        # Get fake tasks for calendar gaps
        fake_tasks_intervals = [
            (
                self.cp_model.NewFixedSizeIntervalVar(
                    start=start,
                    size=end - start,
                    name=f"fake_task_{resource}_{i_task}",
                ),
                value,
            )
            for i_task, (start, end, value) in enumerate(
                self.problem.get_fake_tasks(resource=resource)
            )
        ]

        # Separate blocking intervals by mode
        blocking_data = self._blocking_intervals.get(resource, [])
        reservation_blocking = []
        active_blocking = []

        for blocking_entry in blocking_data:
            interval, demand, metadata = blocking_entry
            if metadata.mode == BlockingMode.RESERVATION:
                reservation_blocking.append((interval, demand))
            else:  # ACTIVE
                active_blocking.append((interval, demand))

        # Get resource capacity
        capacity = self.problem.get_resource_max_capacity(resource)

        # CONSTRAINT 1: Tasks + ALL blocking (RESERVATION + ACTIVE) - NO calendar
        # This enforces RESERVATION blocking even during unavailable periods
        intervals_no_calendar = []
        intervals_no_calendar.extend(task_intervals)
        intervals_no_calendar.extend(reservation_blocking)
        intervals_no_calendar.extend(active_blocking)

        intervals_1 = [
            interval
            for interval, demand in intervals_no_calendar
            if not isinstance(demand, int) or demand > 0
        ]
        demands_1 = [
            demand
            for interval, demand in intervals_no_calendar
            if not isinstance(demand, int) or demand > 0
        ]

        if len(intervals_1) > 0:
            if capacity == 1 and all(isinstance(v, int) and v == 1 for v in demands_1):
                if self.use_no_overlap_for_capa_1 or not self.use_cumulative_for_capa_1:
                    self.cp_model.add_no_overlap(intervals_1)
                if self.use_cumulative_for_capa_1:
                    self.cp_model.add_cumulative(
                        intervals=intervals_1, demands=demands_1, capacity=capacity
                    )
            else:
                self.cp_model.add_cumulative(
                    intervals=intervals_1, demands=demands_1, capacity=capacity
                )

        # CONSTRAINT 2: Tasks + ACTIVE blocking + calendar - NO RESERVATION blocking
        # This enforces ACTIVE blocking only during available periods (constrained by calendar)
        if active_blocking or fake_tasks_intervals:
            intervals_with_calendar = []
            intervals_with_calendar.extend(task_intervals)
            intervals_with_calendar.extend(active_blocking)
            intervals_with_calendar.extend(fake_tasks_intervals)

            intervals_2 = [
                interval
                for interval, demand in intervals_with_calendar
                if not isinstance(demand, int) or demand > 0
            ]
            demands_2 = [
                demand
                for interval, demand in intervals_with_calendar
                if not isinstance(demand, int) or demand > 0
            ]

            if len(intervals_2) > 0:
                if capacity == 1 and all(
                    isinstance(v, int) and v == 1 for v in demands_2
                ):
                    if (
                        self.use_no_overlap_for_capa_1
                        or not self.use_cumulative_for_capa_1
                    ):
                        self.cp_model.add_no_overlap(intervals_2)
                    if self.use_cumulative_for_capa_1:
                        self.cp_model.add_cumulative(
                            intervals=intervals_2, demands=demands_2, capacity=capacity
                        )
                else:
                    self.cp_model.add_cumulative(
                        intervals=intervals_2, demands=demands_2, capacity=capacity
                    )

    def create_resource_blocking_constraints(self) -> None:
        """Create all resource blocking interval variables.

        This should be called during model initialization, before cumulative
        resource constraints are created (so that create_calendar_resources_constraint
        can check for blocking intervals).
        """
        self.create_entity_intervals()
        for entity in self._starts_entity:
            if isinstance(entity, GroupEntity):
                self.constrain_group_entity_times(entity)
        self.create_flexible_gap_blocking_intervals()
        self.create_span_blocking_intervals()

    def create_calendar_resources_constraint(
        self, resource: CumulativeResource
    ) -> None:
        """Create calendar resource constraint, using blocking-aware version if needed.

        Overrides the parent method to automatically use blocking-aware cumulative
        constraints when blocking intervals exist for this resource.

        Args:
            resource: The resource to constrain
        """
        # Check if this resource has blocking constraints
        has_blocking = (
            resource in self._blocking_intervals
            and len(self._blocking_intervals[resource]) > 0
        )

        if has_blocking:
            # Use specialized method that includes blocking intervals
            self.create_cumulative_constraint_including_blocking(resource=resource)
        else:
            # Use standard parent method
            super().create_calendar_resources_constraint(resource=resource)
