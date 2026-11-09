#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Entity-based constraints for flexible scheduling.

This module extends constraint capabilities beyond tasks to any SchedulingEntity.
Entities can be: individual tasks, groups, mode-dependent tasks, hierarchical
compositions, or variable-membership batches.

This allows expressing constraints like:
- "Project (group of tasks) must finish before another task starts" (precedence)
- "Setup period (fixed duration entity) must start after batch ends" (time lag)
- "Batch (variable membership) must start within time window" (time window)
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Generic

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.entities import SchedulingEntity
from discrete_optimization.generic_tasks_tools.enums import MinOrMax, StartOrEnd
from discrete_optimization.generic_tasks_tools.scheduling import SchedulingProblem
from discrete_optimization.generic_tasks_tools.utils import optional_override

logger = logging.getLogger(__name__)


class EntityConstraintsProblem(SchedulingProblem[Task]):
    """Mixin for problems with entity-based constraints.

    This mixin allows defining precedence, time lags, and time windows for
    any SchedulingEntity, not just tasks. This enables more expressive constraints:

    - Precedence between groups: "Phase 1 (group) → Task A"
    - Time lags with mode-dependent entities: "If task in mode 2, wait 5 units before next"
    - Time windows for batches: "Batch must start between 8am and 5pm"
    - Hierarchical constraints: "Department must complete before project starts"

    All methods use @optional_override, so implementing classes only need to
    override methods for constraints they actually use.
    """

    @optional_override
    def get_entity_precedence_constraints(
        self,
    ) -> dict[SchedulingEntity[Task], Iterable[SchedulingEntity[Task]]]:
        """Get precedence constraints between entities.

        Returns:
            Mapping from entity to entities that must follow it.
            entity1 → [entity2, entity3] means entity1.end <= entity2.start AND entity1.end <= entity3.start

        Example:
            >>> # Project phase must complete before individual task
            >>> phase1 = GroupEntity(frozenset({t1, t2, t3}), "phase1")
            >>> task_a = TaskEntity("task_a")
            >>> return {phase1: [task_a]}  # phase1 → task_a

        Default: No entity precedence constraints.
        """
        return {}

    @optional_override
    def get_entity_start_to_start_min_time_lags(
        self,
    ) -> list[tuple[SchedulingEntity[Task], SchedulingEntity[Task], int]]:
        """Get min time lags between entity starts.

        Returns:
            List of (entity1, entity2, offset) where:
            start(entity1) + offset <= start(entity2)

        Default: No constraints.
        """
        return []

    @optional_override
    def get_entity_start_to_start_max_time_lags(
        self,
    ) -> list[tuple[SchedulingEntity[Task], SchedulingEntity[Task], int]]:
        """Get max time lags between entity starts.

        Returns:
            List of (entity1, entity2, offset) where:
            start(entity1) + offset >= start(entity2)

        Default: No constraints.
        """
        return []

    @optional_override
    def get_entity_end_to_start_min_time_lags(
        self,
    ) -> list[tuple[SchedulingEntity[Task], SchedulingEntity[Task], int]]:
        """Get min time lags between entity end and start.

        Returns:
            List of (entity1, entity2, offset) where:
            end(entity1) + offset <= start(entity2)

        Default: No constraints.
        """
        return []

    @optional_override
    def get_entity_end_to_start_max_time_lags(
        self,
    ) -> list[tuple[SchedulingEntity[Task], SchedulingEntity[Task], int]]:
        """Get max time lags between entity end and start.

        Returns:
            List of (entity1, entity2, offset) where:
            end(entity1) + offset >= start(entity2)

        Default: No constraints.
        """
        return []

    @optional_override
    def get_entity_end_to_end_min_time_lags(
        self,
    ) -> list[tuple[SchedulingEntity[Task], SchedulingEntity[Task], int]]:
        """Get min time lags between entity ends.

        Returns:
            List of (entity1, entity2, offset) where:
            end(entity1) + offset <= end(entity2)

        Default: No constraints.
        """
        return []

    @optional_override
    def get_entity_end_to_end_max_time_lags(
        self,
    ) -> list[tuple[SchedulingEntity[Task], SchedulingEntity[Task], int]]:
        """Get max time lags between entity ends.

        Returns:
            List of (entity1, entity2, offset) where:
            end(entity1) + offset >= end(entity2)

        Default: No constraints.
        """
        return []

    @optional_override
    def get_entity_start_to_end_min_time_lags(
        self,
    ) -> list[tuple[SchedulingEntity[Task], SchedulingEntity[Task], int]]:
        """Get min time lags between entity start and end.

        Returns:
            List of (entity1, entity2, offset) where:
            start(entity1) + offset <= end(entity2)

        Default: No constraints.
        """
        return []

    @optional_override
    def get_entity_start_to_end_max_time_lags(
        self,
    ) -> list[tuple[SchedulingEntity[Task], SchedulingEntity[Task], int]]:
        """Get max time lags between entity start and end.

        Returns:
            List of (entity1, entity2, offset) where:
            start(entity1) + offset >= end(entity2)

        Default: No constraints.
        """
        return []

    def get_entity_time_lags(
        self,
        entity1_start_or_end: StartOrEnd,
        entity2_start_or_end: StartOrEnd,
        min_or_max: MinOrMax,
    ) -> list[tuple[SchedulingEntity[Task], SchedulingEntity[Task], int]]:
        """Get entity time lags for specific combination.

        Dispatches to appropriate method based on start/end/min/max combination.

        Args:
            entity1_start_or_end: Whether to use start or end of first entity
            entity2_start_or_end: Whether to use start or end of second entity
            min_or_max: Whether this is a minimum or maximum constraint

        Returns:
            List of (entity1, entity2, offset) tuples
        """
        match entity1_start_or_end, entity2_start_or_end, min_or_max:
            case StartOrEnd.START, StartOrEnd.START, MinOrMax.MIN:
                return self.get_entity_start_to_start_min_time_lags()
            case StartOrEnd.START, StartOrEnd.START, MinOrMax.MAX:
                return self.get_entity_start_to_start_max_time_lags()
            case StartOrEnd.START, StartOrEnd.END, MinOrMax.MIN:
                return self.get_entity_start_to_end_min_time_lags()
            case StartOrEnd.START, StartOrEnd.END, MinOrMax.MAX:
                return self.get_entity_start_to_end_max_time_lags()
            case StartOrEnd.END, StartOrEnd.START, MinOrMax.MIN:
                return self.get_entity_end_to_start_min_time_lags()
            case StartOrEnd.END, StartOrEnd.START, MinOrMax.MAX:
                return self.get_entity_end_to_start_max_time_lags()
            case StartOrEnd.END, StartOrEnd.END, MinOrMax.MIN:
                return self.get_entity_end_to_end_min_time_lags()
            case StartOrEnd.END, StartOrEnd.END, MinOrMax.MAX:
                return self.get_entity_end_to_end_max_time_lags()
            case _:
                return []

    @optional_override
    def get_entity_time_window_lower_bound(
        self, entity: SchedulingEntity[Task], start_or_end: StartOrEnd
    ) -> int | None:
        """Get lower bound on entity start or end time.

        Args:
            entity: The entity to constrain
            start_or_end: Whether constraint applies to start or end

        Returns:
            Lower bound (entity time >= bound), or None if no constraint

        Example:
            >>> batch_entity = VariableBatchEntity("batch_1", candidate_tasks)
            >>> # Batch must start no earlier than time 10
            >>> if entity == batch_entity and start_or_end == StartOrEnd.START:
            ...     return 10

        Default: No lower bound.
        """
        return None

    @optional_override
    def get_entity_time_window_upper_bound(
        self, entity: SchedulingEntity[Task], start_or_end: StartOrEnd
    ) -> int | None:
        """Get upper bound on entity start or end time.

        Args:
            entity: The entity to constrain
            start_or_end: Whether constraint applies to start or end

        Returns:
            Upper bound (entity time <= bound), or None if no constraint

        Example:
            >>> # Batch must complete by time 100
            >>> if entity == batch_entity and start_or_end == StartOrEnd.END:
            ...     return 100

        Default: No upper bound.
        """
        return None


class EntityConstraintsSolution(Generic[Task]):
    """Mixin for solutions to problems with entity constraints.

    Provides checking methods for entity-based constraints.

    This is a mixin class that should be combined with a SchedulingSolution subclass.
    The concrete class must provide:
    - self (SchedulingSolution instance for entity.get_start_time(self), etc.)
    - self.problem (EntityConstraintsProblem instance)
    """

    problem: EntityConstraintsProblem[Task]

    def check_entity_precedence_constraints(self) -> bool:
        """Check that all entity precedence constraints are satisfied.

        Returns:
            True if all constraints satisfied, False otherwise
        """
        for (
            entity1,
            successors,
        ) in self.problem.get_entity_precedence_constraints().items():
            if not entity1.is_active(self):
                continue  # Inactive entities don't enforce constraints

            for entity2 in successors:
                if not entity2.is_active(self):
                    continue  # Skip constraints involving inactive entities

                end1 = entity1.get_end_time(self)
                start2 = entity2.get_start_time(self)

                if end1 > start2:
                    logger.debug(
                        f"Entity precedence violated: {entity1} ends at {end1} "
                        f"but {entity2} starts at {start2}"
                    )
                    return False

        return True

    def check_entity_time_lags(self) -> bool:
        """Check all entity time lag constraints.

        Returns:
            True if all constraints satisfied, False otherwise
        """
        # Check all 8 time lag combinations
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
            for entity1, entity2, offset in self.problem.get_entity_time_lags(
                entity1_se, entity2_se, min_max
            ):
                if not entity1.is_active(self) or not entity2.is_active(self):
                    continue

                # Get appropriate times
                time1 = (
                    entity1.get_start_time(self)
                    if entity1_se == StartOrEnd.START
                    else entity1.get_end_time(self)
                )
                time2 = (
                    entity2.get_start_time(self)
                    if entity2_se == StartOrEnd.START
                    else entity2.get_end_time(self)
                )

                # Check constraint
                if min_max == MinOrMax.MIN:
                    if time1 + offset > time2:
                        logger.debug(
                            f"Entity time lag violated: {entity1} ({entity1_se.value}) "
                            f"at {time1} + {offset} > {entity2} ({entity2_se.value}) at {time2}"
                        )
                        return False
                else:  # MAX
                    if time1 + offset < time2:
                        logger.debug(
                            f"Entity time lag violated: {entity1} ({entity1_se.value}) "
                            f"at {time1} + {offset} < {entity2} ({entity2_se.value}) at {time2}"
                        )
                        return False

        return True

    def check_entity_time_windows(self) -> bool:
        """Check all entity time window constraints.

        Returns:
            True if all constraints satisfied, False otherwise
        """
        # Collect all entities that have constraints
        entities_with_constraints = set()

        # From precedence
        for (
            entity1,
            successors,
        ) in self.problem.get_entity_precedence_constraints().items():
            entities_with_constraints.add(entity1)
            entities_with_constraints.update(successors)

        # From time lags
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
            for entity1, entity2, _ in self.problem.get_entity_time_lags(
                entity1_se, entity2_se, min_max
            ):
                entities_with_constraints.add(entity1)
                entities_with_constraints.add(entity2)

        # Check time windows for all relevant entities
        for entity in entities_with_constraints:
            if not entity.is_active(self):
                continue

            for start_or_end in [StartOrEnd.START, StartOrEnd.END]:
                # Check lower bound
                lb = self.problem.get_entity_time_window_lower_bound(
                    entity, start_or_end
                )
                if lb is not None:
                    actual_time = (
                        entity.get_start_time(self)
                        if start_or_end == StartOrEnd.START
                        else entity.get_end_time(self)
                    )
                    if actual_time < lb:
                        logger.debug(
                            f"Entity time window violated: {entity} {start_or_end.value} "
                            f"at {actual_time} < lower bound {lb}"
                        )
                        return False

                # Check upper bound
                ub = self.problem.get_entity_time_window_upper_bound(
                    entity, start_or_end
                )
                if ub is not None:
                    actual_time = (
                        entity.get_start_time(self)
                        if start_or_end == StartOrEnd.START
                        else entity.get_end_time(self)
                    )
                    if actual_time > ub:
                        logger.debug(
                            f"Entity time window violated: {entity} {start_or_end.value} "
                            f"at {actual_time} > upper bound {ub}"
                        )
                        return False

        return True
