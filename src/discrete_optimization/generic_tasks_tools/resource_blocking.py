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
from discrete_optimization.generic_tasks_tools.multimode import (
    MultimodeSolution,
)
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


@dataclass(frozen=True)
class BlockingConstraintMetadata:
    """Metadata for a blocking constraint.

    Resource blocking intervals are ALWAYS ADDITIVE with task consumption.
    The blocking demand is added on top of task consumption during the blocking period.
    Users should adjust their blocking demands accordingly:
    - If a task uses 2 units and blocking adds 1 unit, total consumption = 3 units
    - Ensure resource capacity can accommodate task + blocking consumption

    Attributes:
        mode: Calendar awareness mode controlling interaction with resource availability:
            - RESERVATION: Blocking can span unavailable periods (nights, weekends).
                          Resource is reserved even when "OFF". Enforced without calendar constraints.
            - ACTIVE: Blocking only during available periods. Resource must be "ON".
                     Enforced with calendar constraints.
        description: Optional human-readable description of the constraint
    """

    mode: BlockingMode = BlockingMode.RESERVATION
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
    MultimodeSolution[Task], Generic[Task, CumulativeResource, OtherCalendarResource]
):
    """
    Mixin for solutions to problems with resource blocking constraints.

    Provides methods to:
    - Compute resource consumption from blocking constraints
    - Check constraint satisfaction with calendar awareness
    - Handle overlap between blocking and task execution

    Should be mixed with SchedulingSolution subclass.
    Inherits from MultimodeSolution to ensure get_mode() is always available.
    """

    problem: ResourceBlockingProblem[Task, CumulativeResource, OtherCalendarResource]

    def compute_blocking_consumption(
        self,
        horizon: int,
        resource: CumulativeResource,
    ) -> np.ndarray:
        """Compute resource consumption from all blocking constraints.

        Blocking is always ADDITIVE: consumption from blocking is added to task consumption.
        This means total resource usage = task consumption + blocking consumption.

        This method:
        1. Computes blocking periods from flexible gap and span constraints
        2. Adds blocking consumption for each period (ADDITIVE behavior)
        3. Validates ACTIVE mode constraints against resource calendar

        Args:
            horizon: Time horizon for the schedule
            resource: The resource to compute consumption for

        Returns:
            Array of length horizon with blocking consumption at each time point.
            Note: This is blocking consumption only. Task consumption is computed separately.
            Total consumption should be verified: task_consumption + blocking_consumption <= capacity

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

            # Apply blocking consumption (ADDITIVE: always adds to consumption)
            consumption[start_time:end_time] += amount

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
    ) -> bool:
        """Validate ACTIVE mode blocking against resource calendar.

        Args:
            resource: The resource being blocked
            start_time: Start of blocking period
            end_time: End of blocking period
            entity1: First entity in constraint
            entity2: Second entity in constraint

        Returns:
            True if valid, False if resource unavailable during blocking period
        """
        # Get calendar from problem
        calendar = self.problem.get_resource_calendar(resource)  # type: ignore

        # Check each time point in blocking period
        for t in range(start_time, end_time):
            if t >= len(calendar):
                break
            if calendar[t] == 0:
                logger.warning(
                    f"ACTIVE blocking constraint violated: resource {resource} "
                    f"is unavailable at time {t}, but blocking from {entity1} to {entity2} "
                    f"spans [{start_time}, {end_time})"
                )
                return False
        return True

    def _validate_active_blocking_span(
        self,
        resource: CumulativeResource,
        start_time: int,
        end_time: int,
        tasks: frozenset[Task],
    ) -> bool:
        """Validate ACTIVE mode span blocking against resource calendar.

        Args:
            resource: The resource being blocked
            start_time: Start of blocking span
            end_time: End of blocking span
            tasks: Tasks defining the span

        Returns:
            True if valid, False if resource unavailable during span
        """
        # Get calendar from problem
        calendar = self.problem.get_resource_calendar(resource)  # type: ignore

        # Check each time point in span
        for t in range(start_time, end_time):
            if t >= len(calendar):
                break
            if calendar[t] == 0:
                logger.warning(
                    f"ACTIVE span blocking constraint violated: resource {resource} "
                    f"is unavailable at time {t}, but span of tasks {tasks} "
                    f"covers [{start_time}, {end_time})"
                )
                return False
        return True

    def check_blocking_constraints(self) -> bool:
        """Check if all blocking constraints are satisfied.

        Mirrors the two-constraint approach from CP-SAT solver:

        Check 1 (RESERVATION constraint - no calendar):
            - Tasks + ALL blocking (RESERVATION + ACTIVE) <= base capacity
            - This allows RESERVATION blocking to span unavailable periods

        Check 2 (ACTIVE constraint - with calendar):
            - Tasks + ACTIVE blocking <= calendar capacity at each time
            - ACTIVE mode: Validate blocking only occurs during available periods

        Returns:
            True if all constraints satisfied, False otherwise
        """
        solution: SchedulingSolution = self  # type: ignore
        horizon = getattr(self.problem, "horizon", 10000)

        # STEP 1: Validate ACTIVE mode calendar constraints
        # ACTIVE blocking can only occur when resource is available
        for (
            entity1,
            point1,
            entity2,
            point2,
            resources,
            metadata,
        ) in self.problem.get_flexible_gap_blocking_constraints():
            if not entity1.is_active(solution) or not entity2.is_active(solution):
                continue

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

            if metadata.mode == BlockingMode.ACTIVE:
                for resource in resources:
                    if not self._validate_active_blocking(
                        resource, start_time, end_time, entity1, entity2
                    ):
                        return False

        # Validate span blocking ACTIVE mode
        for tasks, resources, metadata in self.problem.get_span_blocking_constraints():
            if len(tasks) == 0:
                continue

            start_time = min(solution.get_start_time(t) for t in tasks)
            end_time = max(solution.get_end_time(t) for t in tasks)

            if end_time <= start_time:
                continue

            if metadata.mode == BlockingMode.ACTIVE:
                for resource in resources:
                    if not self._validate_active_blocking_span(
                        resource, start_time, end_time, tasks
                    ):
                        return False

        # STEP 2: Validate capacity constraints (mirroring CP-SAT two-constraint approach)
        # CP-SAT creates TWO constraints:
        # 1. tasks + reservation_blocking + active_blocking <= capacity (no calendar/fake_tasks)
        # 2. tasks + active_blocking + fake_tasks <= capacity (with calendar)
        #
        # Since fake_tasks[t] = capacity - calendar[t], constraint 2 becomes:
        # tasks + active_blocking <= calendar[t]

        for resource in self.problem.cumulative_resources_list:
            capacity = self.problem.get_resource_max_capacity(resource)
            task_consumption = self._compute_task_consumption(resource, horizon)

            # Compute blocking by mode
            reservation_blocking = self._compute_blocking_by_mode(
                resource, horizon, BlockingMode.RESERVATION
            )
            active_blocking = self._compute_blocking_by_mode(
                resource, horizon, BlockingMode.ACTIVE
            )

            # CHECK 1 (mirrors CP-SAT constraint 1):
            # Tasks + RESERVATION blocking + ACTIVE blocking <= base capacity
            # No calendar/fake_tasks - allows RESERVATION to span unavailable periods
            for t in range(horizon):
                total = (
                    task_consumption[t] + reservation_blocking[t] + active_blocking[t]
                )
                if total > capacity:
                    logger.warning(
                        f"Constraint 1 violated for {resource} at time {t}: "
                        f"task={task_consumption[t]} + reservation={reservation_blocking[t]} "
                        f"+ active={active_blocking[t]} = {total} > capacity={capacity}"
                    )
                    return False

            # CHECK 2 (mirrors CP-SAT constraint 2):
            # Tasks + ACTIVE blocking + fake_tasks <= capacity
            # Equivalent to: Tasks + ACTIVE blocking <= calendar[t]
            # This enforces that ACTIVE blocking respects calendar availability
            calendar = self.problem.get_resource_calendar(resource)

            for t in range(min(horizon, len(calendar))):
                calendar_capacity = calendar[t]
                total = task_consumption[t] + active_blocking[t]

                if total > calendar_capacity:
                    logger.warning(
                        f"Constraint 2 violated for {resource} at time {t}: "
                        f"task={task_consumption[t]} + active={active_blocking[t]} "
                        f"= {total} > calendar_capacity={calendar_capacity}"
                    )
                    return False

        return True

    def _compute_task_consumption(
        self, resource: CumulativeResource, horizon: int
    ) -> np.ndarray:
        """Compute resource consumption from tasks.

        Args:
            resource: The resource to compute consumption for
            horizon: Time horizon

        Returns:
            Array of length horizon with task consumption at each time point
        """
        consumption = np.zeros(horizon, dtype=int)

        for task in self.problem.tasks_list:
            start = self.get_start_time(task)
            end = self.get_end_time(task)
            mode = self.get_mode(task)

            # Get consumption amount
            amount = self.problem.get_cumulative_resource_consumption(
                resource, task, mode
            )

            if amount > 0 and start < end:
                consumption[start:end] += amount

        return consumption

    def _compute_blocking_by_mode(
        self, resource: CumulativeResource, horizon: int, mode: BlockingMode
    ) -> np.ndarray:
        """Compute blocking consumption for a specific mode.

        Args:
            resource: The resource to compute consumption for
            horizon: Time horizon
            mode: The blocking mode to filter by (RESERVATION or ACTIVE)

        Returns:
            Array of length horizon with blocking consumption at each time point
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
            # Skip if wrong mode or resource not in constraint
            if metadata.mode != mode or resource not in resources:
                continue

            # Skip inactive entities
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

            # Add blocking consumption
            amount = resources[resource]
            consumption[start_time:end_time] += amount

        # Process span blocking constraints
        for tasks, resources, metadata in self.problem.get_span_blocking_constraints():
            # Skip if wrong mode or resource not in constraint
            if metadata.mode != mode or resource not in resources:
                continue

            if len(tasks) == 0:
                continue

            # Compute span
            start_time = min(solution.get_start_time(t) for t in tasks)
            end_time = max(solution.get_end_time(t) for t in tasks)

            if end_time <= start_time:
                continue

            # Add blocking consumption
            amount = resources[resource]
            consumption[start_time:end_time] += amount

        return consumption

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
