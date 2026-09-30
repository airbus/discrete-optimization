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
from dataclasses import dataclass, field
from enum import Enum
from typing import Generic

import numpy as np

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.cumulative_resource import (
    CumulativeResource,
    CumulativeResourceProblem,
    CumulativeResourceSolution,
    OtherCalendarResource,
)
from discrete_optimization.generic_tasks_tools.entities import SchedulingEntity
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.scheduling import (
    SchedulingSolution,
)
from discrete_optimization.generic_tasks_tools.utils import optional_override

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
        name_choice:
        the name of the underlying choice to be made,
        if choice_resource_blocked is not empty.
    """

    mode: BlockingMode = BlockingMode.RESERVATION
    description: str = ""
    name_choice: str = ""


@dataclass(frozen=True)
class BlockingConstraint(Generic[CumulativeResource]):
    metadata: BlockingConstraintMetadata
    default_resource_blocked: dict[CumulativeResource, int]
    choice_resource_blocked: dict[int, dict[CumulativeResource, int]] = field(
        default_factory=dict
    )

    def has_a_choice(self):
        return len(self.choice_resource_blocked) > 0

    def has_default_resource_blocked(self):
        return len(self.default_resource_blocked) > 0

    def get_all_potential_resources_blocked(self) -> set[CumulativeResource]:
        potential_res = set(self.default_resource_blocked.keys())
        for choice in self.choice_resource_blocked:
            potential_res.update(set(self.choice_resource_blocked[choice]))
        return potential_res


class SpanBlockingConstraint(BlockingConstraint[CumulativeResource]):
    entity: SchedulingEntity


class FlexibleGapBlockingConstraint(BlockingConstraint[CumulativeResource]):
    left_entity: SchedulingEntity
    start_or_end_left_entity: StartOrEnd
    right_entity: SchedulingEntity
    start_or_end_right_entity: StartOrEnd


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

    @optional_override
    def get_flexible_gap_blocking_constraints(
        self,
    ) -> list[FlexibleGapBlockingConstraint[CumulativeResource]]:
        """Return flexible gap blocking constraints.
        Default to no flexible gap blocking constraints.

        """
        return []

    @optional_override
    def get_span_blocking_constraints(
        self,
    ) -> list[SpanBlockingConstraint[CumulativeResource]]:
        """Return span blocking constraints.

        Each constraint blocks resources for the entire span of a task group:
        - From: minimum start time of any task in group
        - To: maximum end time of any task in group

        Returns:
            List of tuples (tasks, resources, metadata):
            - tasks: Frozen set of tasks defining the span
            - resources: Dict mapping resources to consumption amounts
            - metadata: Blocking behavior configuration

        Default to no span blocking constraints.
        """
        return []

    def has_any_blocking(self):
        return (
            len(self.get_flexible_gap_blocking_constraints()) > 0
            or len(self.get_span_blocking_constraints()) > 0
        )


class ResourceBlockingSolution(
    CumulativeResourceSolution[Task, CumulativeResource, OtherCalendarResource],
    Generic[Task, CumulativeResource, OtherCalendarResource],
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

    @optional_override
    def get_choice_of_resource_blocking(
        self,
        blocking_constraint: SpanBlockingConstraint | FlexibleGapBlockingConstraint,
    ) -> int | None:
        if not blocking_constraint.has_a_choice():
            return None
        # Should be stored in the solution object somehow
        raise NotImplementedError

    def is_resource_blocking_active(
        self, resource_blocking_constraint: BlockingConstraint[CumulativeResource]
    ) -> bool:
        if isinstance(resource_blocking_constraint, SpanBlockingConstraint):
            return resource_blocking_constraint.entity.is_active(self)
        if isinstance(resource_blocking_constraint, FlexibleGapBlockingConstraint):
            return resource_blocking_constraint.left_entity.is_active(
                self
            ) and resource_blocking_constraint.right_entity.is_active(self)
        # Other kind of resource blocking constraint not yet implemented.
        raise NotImplementedError

    def get_resource_blocked(
        self,
        resource_blocking_constraint: BlockingConstraint[CumulativeResource],
        resource: CumulativeResource,
    ) -> int:
        active = self.is_resource_blocking_active(resource_blocking_constraint)
        if not active:
            return 0
        amount = 0
        if resource in resource_blocking_constraint.default_resource_blocked:
            amount += resource_blocking_constraint.default_resource_blocked[resource]
        if resource_blocking_constraint.has_a_choice():
            choice = self.get_choice_of_resource_blocking(resource_blocking_constraint)
            if resource in (
                choice_resource := resource_blocking_constraint.choice_resource_blocked[
                    choice
                ]
            ):
                amount += choice_resource[resource]
        return amount

    def get_all_resource_blocked(
        self, resource_blocking_constraint: BlockingConstraint[CumulativeResource]
    ) -> dict[CumulativeResource, int]:
        potential_resource_blocked: set[CumulativeResource] = (
            resource_blocking_constraint.get_all_potential_resources_blocked()
        )
        dict_resource_blocked: dict[CumulativeResource, int] = {}
        for res in potential_resource_blocked:
            amount = self.get_resource_blocked(resource_blocking_constraint, res)
            if amount > 0:
                dict_resource_blocked[res] = amount
        return dict_resource_blocked

    def compute_blocking_consumption(
        self,
        horizon: int,
        resource: CumulativeResource,
        blocking_modes: set[BlockingMode] = None,
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
            blocking_modes: blocking modes to consider
        Returns:
            Array of length horizon with blocking consumption at each time point.
            Note: This is blocking consumption only. Task consumption is computed separately.
            Total consumption should be verified: task_consumption + blocking_consumption <= capacity

        Raises:
            ValueError: If ACTIVE mode blocking spans resource unavailable period
        """
        if blocking_modes is None:
            blocking_modes = {BlockingMode.RESERVATION, BlockingMode.ACTIVE}
        consumption = np.zeros(horizon, dtype=int)
        solution: SchedulingSolution = self  # type: ignore

        # Process flexible gap blocking constraints
        for (
            flexible_gap_blocking
        ) in self.problem.get_flexible_gap_blocking_constraints():
            entity_1 = flexible_gap_blocking.left_entity
            start_or_end_entity_1 = flexible_gap_blocking.start_or_end_left_entity
            entity_2 = flexible_gap_blocking.right_entity
            start_or_end_entity_2 = flexible_gap_blocking.start_or_end_right_entity
            metadata = flexible_gap_blocking.metadata
            if metadata.mode not in blocking_modes:
                continue
            resource_consumption = self.get_resource_blocked(
                flexible_gap_blocking, resource
            )
            if resource_consumption == 0:
                continue
            # Get blocking period
            start_time = (
                entity_1.get_start_time(solution)
                if start_or_end_entity_1 == StartOrEnd.START
                else entity_1.get_end_time(solution)
            )
            end_time = (
                entity_2.get_start_time(solution)
                if start_or_end_entity_2 == StartOrEnd.START
                else entity_2.get_end_time(solution)
            )

            # Skip if blocking period is empty or negative
            if end_time <= start_time:
                continue

            # Get consumption amount
            amount = resource_consumption
            # Validate ACTIVE mode: resource must be available during blocking
            if metadata.mode == BlockingMode.ACTIVE:
                self._validate_active_blocking(
                    resource, start_time, end_time, entity_1, entity_2
                )

            # Apply blocking consumption (ADDITIVE: always adds to consumption)
            consumption[start_time:end_time] += amount

        # Process span blocking constraints
        for span_blocking_constraint in self.problem.get_span_blocking_constraints():
            metadata = span_blocking_constraint.metadata
            if metadata.mode not in blocking_modes:
                continue
            entity = span_blocking_constraint.entity
            resource_consumption = self.get_resource_blocked(
                span_blocking_constraint, resource
            )
            if resource_consumption == 0:
                continue
            # Compute span: min start to max end of all tasks
            start_time: int = entity.get_start_time(solution)
            end_time: int = entity.get_end_time(solution)

            if end_time <= start_time:
                continue

            amount = resource_consumption

            # Validate ACTIVE mode
            if metadata.mode == BlockingMode.ACTIVE:
                self._validate_active_blocking_span(
                    resource, start_time, end_time, entity.get_tasks()
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
        if not self.problem.has_any_blocking():
            return True
        solution: SchedulingSolution = self  # type: ignore
        horizon = self.get_max_end_time()
        # STEP 1: Validate ACTIVE mode calendar constraints
        # ACTIVE blocking can only occur when resource is available
        for (
            flexible_gap_blocking
        ) in self.problem.get_flexible_gap_blocking_constraints():
            if not self.is_resource_blocking_active(flexible_gap_blocking):
                continue
            entity1 = flexible_gap_blocking.left_entity
            entity2 = flexible_gap_blocking.right_entity
            start_or_end_entity1 = flexible_gap_blocking.start_or_end_left_entity
            start_or_end_entity2 = flexible_gap_blocking.start_or_end_right_entity
            metadata = flexible_gap_blocking.metadata
            start_time = (
                entity1.get_start_time(solution)
                if start_or_end_entity1 == StartOrEnd.START
                else entity1.get_end_time(solution)
            )
            end_time = (
                entity2.get_start_time(solution)
                if start_or_end_entity2 == StartOrEnd.START
                else entity2.get_end_time(solution)
            )

            if end_time <= start_time:
                continue

            if metadata.mode == BlockingMode.ACTIVE:
                # Non-zeros resource blocked.
                resource_blocked = self.get_all_resource_blocked(flexible_gap_blocking)
                for resource in resource_blocked:
                    if not self._validate_active_blocking(
                        resource, start_time, end_time, entity1, entity2
                    ):
                        return False

        # Validate span blocking ACTIVE mode
        for span_blocking_constraint in self.problem.get_span_blocking_constraints():
            entity = span_blocking_constraint.entity
            metadata = span_blocking_constraint.metadata
            if len(entity.get_tasks()) == 0:
                continue
            tasks = entity.get_tasks()
            start_time = entity.get_start_time(self)
            end_time = entity.get_end_time(self)
            if end_time <= start_time:
                continue
            if metadata.mode == BlockingMode.ACTIVE:
                resource_blocked = self.get_all_resource_blocked(
                    span_blocking_constraint
                )
                for resource in resource_blocked:
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
            task_consumption = self._compute_calendar_resource_consumption_np(
                {resource}
            )[resource]
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
        return self.compute_blocking_consumption(
            resource=resource, horizon=horizon, blocking_modes={mode}
        )
