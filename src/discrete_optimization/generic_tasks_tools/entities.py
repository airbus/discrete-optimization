#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Entity abstraction for scheduling constraints.

This module provides a unified abstraction for scheduling entities that can participate
in various types of constraints (precedence, time lags, resource blocking, etc.).

An entity represents any schedulable object with start and end times:
- Individual tasks (TaskEntity)
- Groups of tasks (GroupEntity)
- Tasks in specific modes (TaskModeEntity)
- Hierarchical compositions of entities (CompositeEntity)

The abstraction is recursive: CompositeEntity can contain any other entities (including
other CompositeEntity instances), enabling arbitrary hierarchical structures for
constraint modeling (e.g., projects → phases → tasks).

Entities are immutable (frozen dataclasses) so they can be used as dict keys.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Generic

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.scheduling import SchedulingSolution


@dataclass(frozen=True)
class SchedulingEntity(Generic[Task]):
    """Abstract representation of a scheduling entity that has start and end times.

    An entity represents any schedulable object that can participate in constraints:
    - Individual tasks (TaskEntity)
    - Groups of tasks (GroupEntity)
    - Tasks in specific execution modes (TaskModeEntity)
    - Hierarchical compositions of entities (CompositeEntity)
    - Other aggregations (resources, shifts, projects, etc.)

    Entities are immutable (frozen dataclass) so they can be used as dict keys.

    The entity abstraction is recursive: CompositeEntity can contain other entities,
    enabling hierarchical constraint modeling (e.g., projects → phases → tasks).

    The entity abstraction allows expressing complex constraints naturally:
    - "Group of tasks must finish before another task starts" (precedence)
    - "Resource blocked from end of group to start of task" (resource blocking)
    - "If task is in mode 2, then block resource X" (conditional blocking)
    - "All phases of a project must respect a resource limit" (hierarchical constraints)

    Example:
        >>> from discrete_optimization.generic_tasks_tools.entities import TaskEntity, GroupEntity
        >>> # Individual task
        >>> task_ent = TaskEntity("paint")
        >>> # Group of tasks
        >>> group_ent = GroupEntity(frozenset({"prep", "paint", "dry"}), "painting_job")
        >>> # Query in solution
        >>> start = task_ent.get_start_time(solution) # doctest: +SKIP
        >>> group_start = group_ent.get_start_time(solution)  # min of task starts # doctest: +SKIP

    """

    @abstractmethod
    def get_start_time(self, solution: SchedulingSolution) -> int:
        """Get the start time of this entity in the given solution.

        For tasks: the task's start time
        For groups: minimum start time of tasks in group
        For conditional entities: start time if active, else raises error

        Args:
            solution: The scheduling solution to query

        Returns:
            Start time (integer)

        Raises:
            ValueError: If entity is not active/present in solution (e.g., TaskModeEntity
                        when task is not in the specified mode)
        """
        ...

    @abstractmethod
    def get_end_time(self, solution: SchedulingSolution) -> int:
        """Get the end time of this entity in the given solution.

        For tasks: the task's end time
        For groups: maximum end time of tasks in group
        For conditional entities: end time if active, else raises error

        Args:
            solution: The scheduling solution to query

        Returns:
            End time (integer)

        Raises:
            ValueError: If entity is not active/present in solution
        """
        ...

    @abstractmethod
    def is_active(self, solution: SchedulingSolution) -> bool:
        """Check if this entity is active/present in the solution.

        For tasks: always True (task is always scheduled)
        For groups: True if any task in group is scheduled
        For conditional entities: True if the condition is satisfied
            (e.g., TaskModeEntity is active only if task is in specified mode)

        Args:
            solution: The scheduling solution to query

        Returns:
            True if entity is active, False otherwise
        """
        ...

    @abstractmethod
    def get_tasks(self) -> frozenset[Task]:
        """Get all tasks that compose this entity.

        For single tasks: {task}
        For groups: all tasks in the group
        For conditional entities: {task}

        Returns:
            Frozen set of tasks
        """
        ...

    @property
    @abstractmethod
    def entity_id(self) -> Hashable:
        """Unique identifier for this entity.

        Used for hashing, equality, and display.

        Returns:
            Hashable identifier (str, int, tuple, etc.)
        """
        ...

    def __hash__(self) -> int:
        """Entities are hashable (needed for dict keys)."""
        return hash(self.entity_id)

    def __eq__(self, other: object) -> bool:
        """Entities are comparable by their ID."""
        if not isinstance(other, SchedulingEntity):
            return False
        return self.entity_id == other.entity_id

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"{self.__class__.__name__}({self.entity_id})"


@dataclass(frozen=True)
class TaskEntity(SchedulingEntity[Task]):
    """Entity representing a single task.

    This is the most common entity type, wrapping a Task reference.

    Attributes:
        task: The task this entity represents

    Examples:
        >>> entity = TaskEntity(task="assembly")
        >>> entity.get_start_time(solution) # doctest: +SKIP
        >>> entity.get_end_time(solution) # doctest: +SKIP
        >>> entity.get_tasks()
        frozenset({'assembly'})
        >>> entity.is_active(solution) # doctest: +SKIP
    """

    task: Task

    def get_start_time(self, solution: SchedulingSolution) -> int:
        return solution.get_start_time(self.task)

    def get_end_time(self, solution: SchedulingSolution) -> int:
        return solution.get_end_time(self.task)

    def is_active(self, solution: SchedulingSolution) -> bool:
        # Task entities are always active (task is always scheduled)
        return True

    def get_tasks(self) -> frozenset[Task]:
        return frozenset({self.task})

    @property
    def entity_id(self) -> Hashable:
        return ("task", self.task)


@dataclass(frozen=True)
class GroupEntity(SchedulingEntity[Task]):
    """Entity representing a group/batch of tasks.

    The group's start time is the minimum start of its tasks.
    The group's end time is the maximum end of its tasks.

    This is useful for:
    - Representing projects with multiple tasks
    - Modeling batches that must stay together
    - Defining spans that consume resources

    Attributes:
        tasks: Set of tasks in the group (must be non-empty)
        group_id: Optional identifier for the group (for display/debugging)

    Examples:
        >>> entity = GroupEntity(
        ...     tasks=frozenset({"prep", "main", "cleanup"}),
        ...     group_id="maintenance_job_1"
        ... )
        >>> entity.get_start_time(solution)  # min(start of prep, main, cleanup)  # doctest: +SKIP
        >>> entity.get_end_time(solution)    # max(end of prep, main, cleanup) # doctest: +SKIP
        >>> entity.get_tasks()  # doctest: +SKIP
        frozenset({'prep', 'main', 'cleanup'})

    """

    tasks: frozenset[Task]
    group_id: Hashable | None = None

    def __post_init__(self) -> None:
        if len(self.tasks) == 0:
            raise ValueError("GroupEntity must contain at least one task")

    def get_start_time(self, solution: SchedulingSolution) -> int:
        return min(solution.get_start_time(task) for task in self.tasks)

    def get_end_time(self, solution: SchedulingSolution) -> int:
        return max(solution.get_end_time(task) for task in self.tasks)

    def is_active(self, solution: SchedulingSolution) -> bool:
        # Group is active if any task is scheduled
        # (In practice, all tasks should be scheduled)
        return True

    def get_tasks(self) -> frozenset[Task]:
        return self.tasks

    @property
    def entity_id(self) -> Hashable:
        if self.group_id is not None:
            return ("group", self.group_id)
        else:
            # Use sorted tuple of tasks for deterministic ID
            return ("group", tuple(sorted(self.tasks, key=str)))


@dataclass(frozen=True)
class TaskModeEntity(SchedulingEntity[Task]):
    """Entity representing a task in a specific execution mode.

    This entity is only "active" if the task is executed in the specified mode.
    Useful for mode-dependent constraints:
    - "If task A is in mode 1, then block resource X"
    - "Task B can only start after task A in mode 2 completes"

    Attributes:
        task: The task
        mode: The specific mode (integer)

    Examples:
        >>> entity = TaskModeEntity(task="painting", mode=2)
        >>> entity.is_active(solution)  # True only if painting is in mode 2 # doctest: +SKIP
        False  # (if painting is in mode 1)
        >>> # If active:
        >>> entity.get_start_time(solution) # doctest: +SKIP
        10
        >>> entity.get_tasks()
        frozenset({'painting'})

    Raises:
        ValueError: When calling get_start_time() or get_end_time() on an inactive entity

    """

    task: Task
    mode: int

    def get_start_time(self, solution: SchedulingSolution) -> int:
        if not self.is_active(solution):
            raise ValueError(
                f"TaskModeEntity({self.task}, mode={self.mode}) is not active in solution "
                f"(task is in mode {solution.get_mode(self.task)})"
            )
        return solution.get_start_time(self.task)

    def get_end_time(self, solution: SchedulingSolution) -> int:
        if not self.is_active(solution):
            raise ValueError(
                f"TaskModeEntity({self.task}, mode={self.mode}) is not active in solution "
                f"(task is in mode {solution.get_mode(self.task)})"
            )
        return solution.get_end_time(self.task)

    def is_active(self, solution: SchedulingSolution) -> bool:
        """Check if the task is executed in the specified mode."""
        from discrete_optimization.generic_tasks_tools.multimode import (
            MultimodeSolution,
        )

        if not isinstance(solution, MultimodeSolution):
            # Solution doesn't support modes
            return False

        try:
            actual_mode = solution.get_mode(self.task)
            return actual_mode == self.mode
        except (KeyError, AttributeError):
            # Task doesn't have mode information
            return False

    def get_tasks(self) -> frozenset[Task]:
        return frozenset({self.task})

    @property
    def entity_id(self) -> Hashable:
        return ("task_mode", self.task, self.mode)


@dataclass(frozen=True)
class CompositeEntity(SchedulingEntity[Task]):
    """Entity representing a hierarchical composition of other entities.

    This enables recursive entity structures for hierarchical constraint modeling:
    - Groups of groups (e.g., projects containing sub-projects)
    - Mixed collections of tasks, groups, and mode-specific entities
    - Arbitrarily nested entity hierarchies (e.g., departments → teams → tasks)

    The composite's start time is the minimum start of its active children.
    The composite's end time is the maximum end of its active children.

    Examples:
        # Hierarchy: Project → Phases → Tasks
        phase1 = CompositeEntity(
            entities=frozenset({TaskEntity(t1), TaskEntity(t2)}),
            composite_id="phase1"
        )
        phase2 = CompositeEntity(
            entities=frozenset({TaskEntity(t3), TaskEntity(t4)}),
            composite_id="phase2"
        )
        project = CompositeEntity(
            entities=frozenset({phase1, phase2}),
            composite_id="project_alpha"
        )

        # Mixed: combining different entity types
        mixed = CompositeEntity(
            entities=frozenset({
                TaskEntity(t1),
                GroupEntity(tasks=frozenset({t2, t3})),
                TaskModeEntity(task=t4, mode=2)
            })
        )

    Attributes:
        entities: Set of child entities (must be non-empty)
        composite_id: Optional identifier for display/debugging

    Raises:
        ValueError: If entities set is empty, or if all children are inactive when
                   querying start/end times
    """

    entities: frozenset[SchedulingEntity]
    composite_id: Hashable | None = None

    def __post_init__(self) -> None:
        if len(self.entities) == 0:
            raise ValueError("CompositeEntity must contain at least one entity")

    def get_start_time(self, solution: SchedulingSolution) -> int:
        active_entities = [e for e in self.entities if e.is_active(solution)]
        if not active_entities:
            raise ValueError(
                f"CompositeEntity({self.composite_id}) has no active children in solution"
            )
        return min(e.get_start_time(solution) for e in active_entities)

    def get_end_time(self, solution: SchedulingSolution) -> int:
        active_entities = [e for e in self.entities if e.is_active(solution)]
        if not active_entities:
            raise ValueError(
                f"CompositeEntity({self.composite_id}) has no active children in solution"
            )
        return max(e.get_end_time(solution) for e in active_entities)

    def is_active(self, solution: SchedulingSolution) -> bool:
        # Composite is active if at least one child is active
        return any(e.is_active(solution) for e in self.entities)

    def get_tasks(self) -> frozenset[Task]:
        # Recursively collect all tasks from children
        all_tasks = set()
        for entity in self.entities:
            all_tasks.update(entity.get_tasks())
        return frozenset(all_tasks)

    @property
    def entity_id(self) -> Hashable:
        if self.composite_id is not None:
            return ("composite", self.composite_id)
        else:
            # Use sorted tuple of child entity IDs for deterministic ID
            return (
                "composite",
                tuple(sorted((e.entity_id for e in self.entities), key=str)),
            )
