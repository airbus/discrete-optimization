#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any, Generic

from ortools.sat.python.cp_model import IntervalVar, LinearExprT

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.entities import (
    GroupEntity,
    SchedulingEntity,
    TaskEntity,
    TaskModeEntity,
)
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.resource_blocking import (
    BlockingMode,
    CumulativeResource,
    ResourceBlockingProblem,
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

    problem: ResourceBlockingProblem[Task, CumulativeResource, OtherCalendarResource]
    _blocking_intervals: dict

    def init_model(self, **kwargs: Any) -> None:
        """Initialize model and reset blocking interval storage."""
        super().init_model(**kwargs)
        self._blocking_intervals: dict[
            CumulativeResource, list[tuple[IntervalVar, int]]
        ] = {}

    def get_entity_start_variable(self, entity: SchedulingEntity) -> LinearExprT:
        """Get the start time variable for a scheduling entity.

        Args:
            entity: The scheduling entity (TaskEntity, GroupEntity, etc.)

        Returns:
            LinearExprT representing the start time
        """
        if isinstance(entity, TaskEntity):
            return self.get_task_start_or_end_variable(
                task=entity.task, start_or_end=StartOrEnd.START
            )
        elif isinstance(entity, GroupEntity):
            # For groups, start is min of task starts
            return self.cp_model.NewIntVar(
                lb=0,
                ub=self.get_makespan_upper_bound(),
                name=f"group_start_{entity.entity_id}",
            )
        elif isinstance(entity, TaskModeEntity):
            return self.get_task_start_or_end_variable(
                task=entity.task, start_or_end=StartOrEnd.START
            )
        else:
            raise NotImplementedError(f"Unsupported entity type: {type(entity)}")

    def get_entity_end_variable(self, entity: SchedulingEntity) -> LinearExprT:
        """Get the end time variable for a scheduling entity.

        Args:
            entity: The scheduling entity (TaskEntity, GroupEntity, etc.)

        Returns:
            LinearExprT representing the end time
        """
        if isinstance(entity, TaskEntity):
            return self.get_task_start_or_end_variable(
                task=entity.task, start_or_end=StartOrEnd.END
            )
        elif isinstance(entity, GroupEntity):
            # For groups, end is max of task ends
            return self.cp_model.NewIntVar(
                lb=0,
                ub=self.get_makespan_upper_bound(),
                name=f"group_end_{entity.entity_id}",
            )
        elif isinstance(entity, TaskModeEntity):
            return self.get_task_start_or_end_variable(
                task=entity.task, start_or_end=StartOrEnd.END
            )
        else:
            raise NotImplementedError(f"Unsupported entity type: {type(entity)}")

    def get_entity_time_variable(
        self, entity: SchedulingEntity, start_or_end: StartOrEnd
    ) -> LinearExprT:
        """Get start or end time variable for an entity.

        Args:
            entity: The scheduling entity
            start_or_end: Whether to get start or end time

        Returns:
            LinearExprT representing the time
        """
        if start_or_end == StartOrEnd.START:
            return self.get_entity_start_variable(entity)
        else:
            return self.get_entity_end_variable(entity)

    def constrain_group_entity_times(self, entity: GroupEntity) -> None:
        """Add constraints for group entity start/end times.

        A group's start is the minimum start of its tasks.
        A group's end is the maximum end of its tasks.

        Args:
            entity: The group entity
        """
        group_start = self.get_entity_start_variable(entity)
        group_end = self.get_entity_end_variable(entity)

        # group_start <= min(task starts)
        for task in entity.tasks:
            task_start = self.get_task_start_or_end_variable(
                task=task, start_or_end=StartOrEnd.START
            )
            self.cp_model.Add(group_start <= task_start)

        # group_end >= max(task ends)
        for task in entity.tasks:
            task_end = self.get_task_start_or_end_variable(
                task=task, start_or_end=StartOrEnd.END
            )
            self.cp_model.Add(group_end >= task_end)

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

            # Handle group entities
            if isinstance(entity1, GroupEntity):
                self.constrain_group_entity_times(entity1)
            if isinstance(entity2, GroupEntity):
                self.constrain_group_entity_times(entity2)

            # Get time variables for the gap boundaries
            gap_start = self.get_entity_time_variable(entity1, ref1)
            gap_end = self.get_entity_time_variable(entity2, ref2)

            # Create a variable for the gap size
            # The gap may be 0 or positive (we'll enforce positive later)
            gap_size = self.cp_model.NewIntVar(
                lb=0,
                ub=self.get_makespan_upper_bound(),
                name=f"gap_size_{i_constraint}",
            )

            # Create a boolean variable for gap validity (positive duration)
            gap_is_present = self.cp_model.NewBoolVar(f"gap_valid_{i_constraint}")

            # Constrain gap_size = gap_end - gap_start when gap is present
            self.cp_model.Add(gap_size == gap_end - gap_start).OnlyEnforceIf(
                gap_is_present
            )

            # Constraint: gap is only present if gap_end > gap_start (positive duration)
            self.cp_model.Add(gap_end > gap_start).OnlyEnforceIf(gap_is_present)
            self.cp_model.Add(gap_end <= gap_start).OnlyEnforceIf(gap_is_present.Not())

            # Create interval variable for the gap
            gap_interval = self.cp_model.NewOptionalIntervalVar(
                start=gap_start,
                size=gap_size,
                end=gap_end,
                is_present=gap_is_present,
                name=f"blocking_gap_{entity1.entity_id}_{ref1}_to_{entity2.entity_id}_{ref2}",
            )

            # For TaskModeEntity, only block when task is in the specified mode
            if isinstance(entity1, TaskModeEntity):
                mode_present = self.get_task_mode_is_present_variable(
                    task=entity1.task, mode=entity1.mode
                )
                self.cp_model.AddImplication(gap_is_present, mode_present)

            if isinstance(entity2, TaskModeEntity):
                mode_present = self.get_task_mode_is_present_variable(
                    task=entity2.task, mode=entity2.mode
                )
                self.cp_model.AddImplication(gap_is_present, mode_present)

            # Store blocking intervals per resource
            for resource, demand in resources.items():
                if resource not in self._blocking_intervals:
                    self._blocking_intervals[resource] = []

                # Handle blocking mode
                if metadata.mode == BlockingMode.ACTIVE:
                    # Active mode: only block when entity1 is active
                    # This is already handled by the gap_interval presence
                    pass

                self._blocking_intervals[resource].append((gap_interval, demand))

    def create_span_blocking_intervals(self) -> None:
        """Create interval variables for span blocking constraints.

        For each span constraint, creates an interval from the minimum start time
        to the maximum end time of the specified tasks, with the specified resource demands.
        """
        for i_constraint, constraint in enumerate(
            self.problem.get_span_blocking_constraints()
        ):
            tasks, resources, metadata = constraint

            # Create variables for span start and end
            span_start = self.cp_model.NewIntVar(
                lb=0,
                ub=self.get_makespan_upper_bound(),
                name=f"span_start_{i_constraint}",
            )
            span_end = self.cp_model.NewIntVar(
                lb=0,
                ub=self.get_makespan_upper_bound(),
                name=f"span_end_{i_constraint}",
            )

            # Create a variable for the span size
            span_size = self.cp_model.NewIntVar(
                lb=0,
                ub=self.get_makespan_upper_bound(),
                name=f"span_size_{i_constraint}",
            )

            # Constrain span_size = span_end - span_start
            self.cp_model.Add(span_size == span_end - span_start)

            # span_start = min(task starts)
            for task in tasks:
                task_start = self.get_task_start_or_end_variable(
                    task=task, start_or_end=StartOrEnd.START
                )
                self.cp_model.Add(span_start <= task_start)

            # span_end = max(task ends)
            for task in tasks:
                task_end = self.get_task_start_or_end_variable(
                    task=task, start_or_end=StartOrEnd.END
                )
                self.cp_model.Add(span_end >= task_end)

            # Create interval for the span
            span_interval = self.cp_model.NewIntervalVar(
                start=span_start,
                size=span_size,
                end=span_end,
                name=f"blocking_span_{i_constraint}",
            )

            # Store blocking intervals per resource
            for resource, demand in resources.items():
                if resource not in self._blocking_intervals:
                    self._blocking_intervals[resource] = []
                self._blocking_intervals[resource].append((span_interval, demand))

    def get_blocking_intervals(
        self, resource: CumulativeResource
    ) -> list[tuple[IntervalVar, int]]:
        """Get all blocking intervals for a given resource.

        Args:
            resource: The resource

        Returns:
            List of (interval, demand) tuples for blocking constraints
        """
        return self._blocking_intervals.get(resource, [])

    def create_cumulative_constraint_including_blocking(
        self, resource: CumulativeResource
    ) -> None:
        """

        Args:
            resource: The cumulative resource to constrain
        """
        # Get task consumption intervals
        task_intervals_and_demands = super().get_resource_consumption_intervals(
            resource
        )

        # Get fake tasks for calendar gaps
        fake_tasks_intervals_and_demands = [
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

        # Get blocking intervals for this resource
        blocking_intervals_and_demands = self.get_blocking_intervals(resource)

        # Combine ALL intervals (tasks + fake tasks + blocking)
        all_intervals_and_demands = (
            task_intervals_and_demands
            + fake_tasks_intervals_and_demands
            + blocking_intervals_and_demands
        )

        # Filter out zero-demand intervals
        intervals = [
            interval
            for interval, demand in all_intervals_and_demands
            if not isinstance(demand, int) or demand > 0
        ]
        demands = [
            demand
            for interval, demand in all_intervals_and_demands
            if not isinstance(demand, int) or demand > 0
        ]

        # Get resource capacity
        capacity = self.problem.get_resource_max_capacity(resource)

        # Create THE cumulative constraint (replaces standard constraint for this resource)
        if len(intervals) > 0:
            if capacity == 1 and all(
                isinstance(value, int) and value == 1 for value in demands
            ):
                # Special case: capacity 1 with unit demands
                if self.use_no_overlap_for_capa_1 or not self.use_cumulative_for_capa_1:
                    self.cp_model.add_no_overlap(intervals)
                if self.use_cumulative_for_capa_1:
                    self.cp_model.add_cumulative(
                        intervals=intervals,
                        demands=demands,
                        capacity=capacity,
                    )
            else:
                # General case
                self.cp_model.add_cumulative(
                    intervals=intervals,
                    demands=demands,
                    capacity=capacity,
                )

    def create_resource_blocking_constraints(self) -> None:
        """Create all resource blocking interval variables.

        This should be called during model initialization, before cumulative
        resource constraints are created (so that create_calendar_resources_constraint
        can check for blocking intervals).
        """
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
