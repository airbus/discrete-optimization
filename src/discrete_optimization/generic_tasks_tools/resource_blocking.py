#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Resource blocking constraints for scheduling problems.

This module provides mixins for modeling resource blocking during non-execution periods:
- Gap blocking: Resources blocked between two entities (e.g., changeover time)
- Span blocking: Resources blocked for entire span of task group (e.g., project reservation)

Key features:
- Flexible blocking points: START/END of entities (tasks, groups, conditional)
- Calendar awareness: RESERVATION (spans unavailable periods) vs ACTIVE (must be available)
- Overlap handling: Strategies to avoid double-counting when tasks overlap with blocking
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass
from enum import Enum
from typing import Generic

import numpy as np

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.cumulative_resource import (
    CumulativeResource,
    CumulativeResourceProblem,
    OtherCalendarResource,
)
from discrete_optimization.generic_tasks_tools.entities import SchedulingEntity
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.scheduling import (
    SchedulingSolution,
)

logger = logging.getLogger(__name__)


class BlockingMode(Enum):
    """Mode for resource blocking behavior.

    Attributes:
        RESERVATION: Resource slot is reserved but doesn't need to be "ON" or available.
            Blocking can span periods when resource is unavailable (nights, weekends).
        ACTIVE: Resource must be available/ON during blocking period.
            Blocking invalid if resource unavailable during any part of the period.
    """

    RESERVATION = "reservation"
    ACTIVE = "active"


class OverlapHandling(Enum):
    """Strategy for handling overlapping resource consumption.

    Attributes:
        EXCLUSIVE: Don't count blocking during task execution (default, safest).
            Total = task consumption (blocking ignored during overlap).
        ADDITIVE: Add task and blocking consumption.
            Total = task consumption + blocking consumption.
        MAXIMUM: Take maximum of task and blocking consumption.
            Total = max(task consumption, blocking consumption).
        SEPARATE: Use different resource names for tasks and blocking.
            No overlap issue - tracked separately.
    """

    EXCLUSIVE = "exclusive"
    ADDITIVE = "additive"
    MAXIMUM = "maximum"
    SEPARATE = "separate"


@dataclass(frozen=True)
class BlockingConstraintMetadata:
    """Metadata for a blocking constraint.

    Attributes:
        mode: Calendar awareness mode (RESERVATION or ACTIVE)
        overlap_handling: Strategy for overlapping consumption
        description: Optional human-readable description of the constraint
    """

    mode: BlockingMode = BlockingMode.RESERVATION
    overlap_handling: OverlapHandling = OverlapHandling.EXCLUSIVE
    description: str = ""


# Type aliases for blocking constraints (defined after BlockingConstraintMetadata)
FlexibleGapBlockingConstraint = tuple[
    SchedulingEntity,
    StartOrEnd,
    SchedulingEntity,
    StartOrEnd,
    dict[Hashable, int],  # resources
    BlockingConstraintMetadata,
]

SpanBlockingConstraint = tuple[
    frozenset[Hashable],  # tasks
    dict[Hashable, int],  # resources
    BlockingConstraintMetadata,
]


class ResourceBlockingProblem(
    CumulativeResourceProblem[Task, CumulativeResource, OtherCalendarResource],
    Generic[Task, CumulativeResource, OtherCalendarResource],
):
    """Mixin for problems with resource blocking constraints.

    This mixin adds support for two types of blocking:
    1. Flexible gap blocking: Block resources from one entity point to another
    2. Span blocking: Block resources for entire span of task group

    The problem should also inherit from CumulativeResourceProblem to provide
    resource definitions and capacity constraints.
    """

    @abstractmethod
    def get_flexible_gap_blocking_constraints(
        self,
    ) -> list[
        tuple[
            SchedulingEntity,
            StartOrEnd,
            SchedulingEntity,
            StartOrEnd,
            dict[CumulativeResource, int],
            BlockingConstraintMetadata,
        ]
    ]:
        """Return flexible gap blocking constraints.

        Each constraint blocks resources from one entity point to another entity point.
        Supports four patterns based on start/end combinations:
        - END → START: Classic gap/changeover (most common)
        - START → START: Preparation period
        - START → END: Full span coverage
        - END → END: Extended cleanup

        Returns:
            List of tuples (entity1, point1, entity2, point2, resources, metadata):
            - entity1: First entity (task, group, or conditional)
            - point1: START or END of first entity
            - entity2: Second entity
            - point2: START or END of second entity
            - resources: Dict mapping resources to consumption amounts
            - metadata: Blocking behavior configuration
        """
        ...

    @abstractmethod
    def get_span_blocking_constraints(
        self,
    ) -> list[
        tuple[
            frozenset[Task], dict[CumulativeResource, int], BlockingConstraintMetadata
        ]
    ]:
        """Return span blocking constraints.

        Each constraint blocks resources for the entire span of a task group:
        - From: minimum start time of any task in group
        - To: maximum end time of any task in group

        Returns:
            List of tuples (tasks, resources, metadata):
            - tasks: Frozen set of tasks defining the span
            - resources: Dict mapping resources to consumption amounts
            - metadata: Blocking behavior configuration

        Examples:
        """
        ...


class ResourceBlockingSolution(
    SchedulingSolution[Task], Generic[Task, CumulativeResource, OtherCalendarResource]
):
    """
    Mixin for solutions to problems with resource blocking constraints.

    Provides methods to:
    - Compute resource consumption from blocking constraints
    - Check constraint satisfaction with calendar awareness
    - Handle overlap between blocking and task execution

    Should be mixed with SchedulingSolution or subclass.
    """

    problem: ResourceBlockingProblem[Task, CumulativeResource, OtherCalendarResource]

    def compute_blocking_consumption(
        self,
        horizon: int,
        resource: CumulativeResource,
    ) -> np.ndarray:
        """Compute resource consumption from all blocking constraints.

        This method:
        1. Computes blocking periods from flexible gap and span constraints
        2. Applies overlap handling strategies when blocking overlaps with tasks
        3. Validates ACTIVE mode constraints against resource calendar

        Args:
            horizon: Time horizon for the schedule
            resource: The resource to compute consumption for

        Returns:
            Array of length horizon with blocking consumption at each time point.
            Note: This is blocking consumption only. Task consumption is computed separately.

        Raises:
            ValueError: If ACTIVE mode blocking spans resource unavailable period
        """
        consumption = np.zeros(horizon, dtype=int)
        solution: SchedulingSolution = self  # type: ignore

        # Process flexible gap blocking constraints
        for (
            entity1,
            point1,
            entity2,
            point2,
            resources,
            metadata,
        ) in self.problem.get_flexible_gap_blocking_constraints():
            # Skip if resource not in this constraint
            if resource not in resources:
                continue

            # Skip if conditional entity is not active
            if not entity1.is_active(solution) or not entity2.is_active(solution):
                continue

            # Get blocking period
            start_time = (
                entity1.get_start_time(solution)
                if point1 == StartOrEnd.START
                else entity1.get_end_time(solution)
            )
            end_time = (
                entity2.get_start_time(solution)
                if point2 == StartOrEnd.START
                else entity2.get_end_time(solution)
            )

            # Skip if blocking period is empty or negative
            if end_time <= start_time:
                continue

            # Get consumption amount
            amount = resources[resource]

            # Validate ACTIVE mode: resource must be available during blocking
            if metadata.mode == BlockingMode.ACTIVE:
                self._validate_active_blocking(
                    resource, start_time, end_time, entity1, entity2
                )

            # Apply blocking consumption based on overlap handling
            self._apply_blocking_consumption(
                consumption,
                start_time,
                end_time,
                amount,
                metadata.overlap_handling,
                entity1,
                entity2,
                resource,
            )

        # Process span blocking constraints
        for tasks, resources, metadata in self.problem.get_span_blocking_constraints():
            # Skip if resource not in this constraint
            if resource not in resources:
                continue

            # Compute span: min start to max end of all tasks
            if len(tasks) == 0:
                continue

            start_time = min(solution.get_start_time(t) for t in tasks)
            end_time = max(solution.get_end_time(t) for t in tasks)

            if end_time <= start_time:
                continue

            amount = resources[resource]

            # Validate ACTIVE mode
            if metadata.mode == BlockingMode.ACTIVE:
                self._validate_active_blocking_span(
                    resource, start_time, end_time, tasks
                )

            # For span blocking, apply consumption directly
            # Overlap handling is less relevant since span covers task execution
            consumption[start_time:end_time] += amount

        return consumption

    def _validate_active_blocking(
        self,
        resource: CumulativeResource,
        start_time: int,
        end_time: int,
        entity1: SchedulingEntity,
        entity2: SchedulingEntity,
    ) -> None:
        """Validate ACTIVE mode blocking against resource calendar.

        Args:
            resource: The resource being blocked
            start_time: Start of blocking period
            end_time: End of blocking period
            entity1: First entity in constraint
            entity2: Second entity in constraint

        Raises:
            ValueError: If resource unavailable during any part of blocking period
        """
        # Get calendar from problem (CalendarResourceProblem provides get_resource_calendar)
        # CumulativeResourceProblem inherits from CalendarResourceProblem
        calendar = self.problem.get_resource_calendar(resource)  # type: ignore

        # Check each time point in blocking period
        for t in range(start_time, end_time):
            if t >= len(calendar):
                break
            if calendar[t] == 0:
                raise ValueError(
                    f"ACTIVE blocking constraint violated: resource {resource} "
                    f"is unavailable at time {t}, but blocking from {entity1} to {entity2} "
                    f"spans [{start_time}, {end_time}). Use BlockingMode.RESERVATION if blocking "
                    f"should continue through unavailable periods."
                )

    def _validate_active_blocking_span(
        self,
        resource: CumulativeResource,
        start_time: int,
        end_time: int,
        tasks: frozenset[Task],
    ) -> None:
        """Validate ACTIVE mode span blocking against resource calendar.

        Args:
            resource: The resource being blocked
            start_time: Start of blocking span
            end_time: End of blocking span
            tasks: Tasks defining the span

        Raises:
            ValueError: If resource unavailable during any part of span
        """
        # Get calendar from problem (CalendarResourceProblem provides get_resource_calendar)
        calendar = self.problem.get_resource_calendar(resource)  # type: ignore

        # Check each time point in span
        for t in range(start_time, end_time):
            if t >= len(calendar):
                break
            if calendar[t] == 0:
                raise ValueError(
                    f"ACTIVE span blocking constraint violated: resource {resource} "
                    f"is unavailable at time {t}, but span of tasks {tasks} "
                    f"covers [{start_time}, {end_time}). Use BlockingMode.RESERVATION if blocking "
                    f"should continue through unavailable periods."
                )

    def _apply_blocking_consumption(
        self,
        consumption: np.ndarray,
        start_time: int,
        end_time: int,
        amount: int,
        overlap_handling: OverlapHandling,
        entity1: SchedulingEntity,
        entity2: SchedulingEntity,
        resource: CumulativeResource,
    ) -> None:
        """Apply blocking consumption based on overlap handling strategy.

        Args:
            consumption: Array to update with blocking consumption
            start_time: Start of blocking period
            end_time: End of blocking period
            amount: Amount of resource blocked
            overlap_handling: Strategy for handling overlaps
            entity1: First entity in constraint
            entity2: Second entity in constraint
            resource: The resource being blocked
        """
        solution: SchedulingSolution = self  # type: ignore

        if overlap_handling == OverlapHandling.EXCLUSIVE:
            # Add blocking only where no tasks from entities are running
            for t in range(start_time, end_time):
                # Check if any task from entity1 or entity2 is running at time t
                running = False
                for task in entity1.get_tasks() | entity2.get_tasks():
                    task_start = solution.get_start_time(task)
                    task_end = solution.get_end_time(task)
                    if task_start <= t < task_end:
                        running = True
                        break

                if not running:
                    consumption[t] += amount

        elif overlap_handling == OverlapHandling.ADDITIVE:
            # Simply add blocking consumption (may cause capacity violation)
            consumption[start_time:end_time] += amount

        elif overlap_handling == OverlapHandling.MAXIMUM:
            # Take maximum of current and blocking consumption
            consumption[start_time:end_time] = np.maximum(
                consumption[start_time:end_time], amount
            )

        elif overlap_handling == OverlapHandling.SEPARATE:
            # SEPARATE means different resource names - no overlap by definition
            # Just add the consumption
            consumption[start_time:end_time] += amount

    def check_blocking_constraints(self) -> bool:
        """Check if all blocking constraints are satisfied.

        This includes:
        - ACTIVE mode constraints: resource available during blocking
        - Capacity constraints: total consumption (tasks + blocking) <= capacity

        Returns:
            True if all constraints satisfied, False otherwise
        """
        solution: SchedulingSolution = self  # type: ignore

        # Get horizon from problem if available
        horizon = getattr(self.problem, "horizon", 10000)

        try:
            # Validate all flexible gap blocking constraints
            for (
                entity1,
                point1,
                entity2,
                point2,
                resources,
                metadata,
            ) in self.problem.get_flexible_gap_blocking_constraints():
                # Skip inactive conditional entities
                if not entity1.is_active(solution) or not entity2.is_active(solution):
                    continue

                # Get blocking period
                start_time = (
                    entity1.get_start_time(solution)
                    if point1 == StartOrEnd.START
                    else entity1.get_end_time(solution)
                )
                end_time = (
                    entity2.get_start_time(solution)
                    if point2 == StartOrEnd.START
                    else entity2.get_end_time(solution)
                )

                if end_time <= start_time:
                    continue

                # Validate ACTIVE mode for each resource
                if metadata.mode == BlockingMode.ACTIVE:
                    for resource in resources:
                        self._validate_active_blocking(
                            resource, start_time, end_time, entity1, entity2
                        )

            # Validate span blocking constraints
            for (
                tasks,
                resources,
                metadata,
            ) in self.problem.get_span_blocking_constraints():
                if len(tasks) == 0:
                    continue

                start_time = min(solution.get_start_time(t) for t in tasks)
                end_time = max(solution.get_end_time(t) for t in tasks)

                if end_time <= start_time:
                    continue

                # Validate ACTIVE mode for each resource
                if metadata.mode == BlockingMode.ACTIVE:
                    for resource in resources:
                        self._validate_active_blocking_span(
                            resource, start_time, end_time, tasks
                        )

            return True

        except ValueError as e:
            logger.warning(f"Blocking constraint check failed: {e}")
            return False

    def satisfy(self) -> bool:
        """Check if solution satisfies all constraints including blocking.

        This extends the base satisfy() method to include blocking constraints.

        Returns:
            True if all constraints satisfied, False otherwise
        """
        # Check base constraints if available
        if hasattr(super(), "satisfy"):
            if not super().satisfy():  # type: ignore
                return False

        # Check blocking constraints
        return self.check_blocking_constraints()


class WithoutResourceBlockingProblem(
    ResourceBlockingProblem[Task, CumulativeResource, OtherCalendarResource]
):
    """Utility mixin for problems without resource blocking constraints.

    Provides empty implementations of blocking constraint methods.
    Use as an additional mixin with GenericSchedulingProblem when no blocking needed.
    """

    def get_flexible_gap_blocking_constraints(
        self,
    ) -> list[
        tuple[
            SchedulingEntity,
            StartOrEnd,
            SchedulingEntity,
            StartOrEnd,
            dict[CumulativeResource, int],
            BlockingConstraintMetadata,
        ]
    ]:
        """Return empty list (no blocking constraints)."""
        return []

    def get_span_blocking_constraints(
        self,
    ) -> list[
        tuple[
            frozenset[Task], dict[CumulativeResource, int], BlockingConstraintMetadata
        ]
    ]:
        """Return empty list (no blocking constraints)."""
        return []


class WithoutResourceBlockingSolution(
    ResourceBlockingSolution[Task, CumulativeResource, OtherCalendarResource]
):
    """Utility mixin for solutions without resource blocking constraints.

    Provides optimized implementations that skip blocking computation.
    Use as an additional mixin when no blocking needed.
    """

    def check_blocking_constraints(self) -> bool:
        """Always satisfied (no constraints)."""
        return True

    def compute_blocking_consumption(
        self,
        horizon: int,
        resource: CumulativeResource,
    ) -> np.ndarray:
        """Return zero consumption (no blocking)."""
        return np.zeros(horizon, dtype=int)
