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
from discrete_optimization.generic_tasks_tools.batching import BatchingSolution
from discrete_optimization.generic_tasks_tools.enums import AbsentValue, StartOrEnd
from discrete_optimization.generic_tasks_tools.scheduling import SchedulingSolution
from discrete_optimization.generic_tasks_tools.utils import (
    optional_override,
    optional_override_implem,
)


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
    def get_start_time(self, solution: SchedulingSolution) -> int | AbsentValue:
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
    def get_end_time(self, solution: SchedulingSolution) -> int | AbsentValue:
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

    @optional_override
    def unfold_entities(self):
        yield

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

    def get_start_time(self, solution: SchedulingSolution) -> int | AbsentValue:
        return solution.get_start_time(self.task)

    def get_end_time(self, solution: SchedulingSolution) -> int | AbsentValue:
        return solution.get_end_time(self.task)

    def is_active(self, solution: SchedulingSolution) -> bool:
        return solution.is_present(self.task)

    def get_tasks(self) -> frozenset[Task]:
        return frozenset({self.task})

    @property
    def entity_id(self) -> Hashable:
        return "task", self.task


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

    def get_start_time(self, solution: SchedulingSolution) -> int | AbsentValue:
        active_tasks = [task for task in self.tasks if solution.is_present(task)]
        if len(active_tasks) == 0:
            return AbsentValue.ABSENT
        else:
            return min(solution.get_start_time(task) for task in active_tasks)

    def get_end_time(self, solution: SchedulingSolution) -> int | AbsentValue:
        active_tasks = [task for task in self.tasks if solution.is_present(task)]
        if len(active_tasks) == 0:
            return AbsentValue.ABSENT
        else:
            return max(solution.get_end_time(task) for task in active_tasks)

    def is_active(self, solution: SchedulingSolution) -> bool:
        # Group is active if any task is scheduled
        return any(task for task in self.tasks if solution.is_present(task))

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

    def get_start_time(self, solution: SchedulingSolution) -> int | AbsentValue.ABSENT:
        if not self.is_active(solution):
            return AbsentValue.ABSENT
        return solution.get_start_time(self.task)

    def get_end_time(self, solution: SchedulingSolution) -> int | AbsentValue.ABSENT:
        if not self.is_active(solution):
            return AbsentValue.ABSENT
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
        except (KeyError, AttributeError):
            # Task doesn't have mode information
            return False
        else:
            return actual_mode == self.mode and solution.is_present(self.task)

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

    def get_start_time(self, solution: SchedulingSolution) -> int | AbsentValue:
        active_entities = [e for e in self.entities if e.is_active(solution)]
        if not active_entities:
            return AbsentValue.ABSENT
        return min(e.get_start_time(solution) for e in active_entities)

    def get_end_time(self, solution: SchedulingSolution) -> int | AbsentValue:
        active_entities = [e for e in self.entities if e.is_active(solution)]
        if not active_entities:
            return AbsentValue.ABSENT
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

    @optional_override_implem
    def unfold_entities(self):
        for entity in self.entities:
            yield entity
            entity.unfold_entities()


@dataclass(frozen=True)
class ConstantDurationEntity(SchedulingEntity[Task]):
    """
    Entity representing
    a constant duration block
    starting at start or end of a given entity
    """

    other_entity: SchedulingEntity[Task]
    constant_duration: int
    offset: int = 0
    start_or_end: StartOrEnd = StartOrEnd.START
    constant_duration_id: Hashable | None = None

    def get_start_time(self, solution: SchedulingSolution) -> int | AbsentValue:
        match self.start_or_end:
            case StartOrEnd.START:
                return self.other_entity.get_start_time(solution) + self.offset
            case StartOrEnd.END:
                return self.other_entity.get_end_time(solution) + self.offset
            case _:
                raise ValueError(f"{self.start_or_end} is not a valid Enum")

    def get_end_time(self, solution: SchedulingSolution) -> int | AbsentValue:
        if self.is_active(solution):
            return self.get_start_time(solution) + self.constant_duration
        return AbsentValue.ABSENT

    def is_active(self, solution: SchedulingSolution) -> bool:
        return self.other_entity.is_active(solution)

    def get_tasks(self) -> frozenset[Task]:
        return self.other_entity.get_tasks()

    @property
    def entity_id(self) -> Hashable:
        if self.constant_duration_id is None:
            return (
                "constant_block",
                self.other_entity.entity_id,
                self.constant_duration,
                self.start_or_end,
            )
        return self.constant_duration_id

    @optional_override_implem
    def unfold_entities(self):
        yield self.other_entity
        self.other_entity.unfold_entities()


@dataclass(frozen=True)
class MultiplyTaskEntity(SchedulingEntity[Task]):
    # Allows to create some interval based on a base one:
    # if task is the interval [2, 3], multiply_factor is 3, and start_or_end = Start
    # Then the entity is [6, 7], if start_or_end = End :
    # will be, [8, 9]
    task: Task
    multiply_factor: int
    start_or_end: StartOrEnd = StartOrEnd.START

    def get_start_time(self, solution: SchedulingSolution) -> int | AbsentValue:
        if not solution.is_present(self.task):
            return AbsentValue.ABSENT
        match self.start_or_end:
            case StartOrEnd.START:
                return solution.get_start_time(self.task) * self.multiply_factor
            case StartOrEnd.END:
                dur = solution.get_duration(self.task)
                return solution.get_end_time(self.task) * self.multiply_factor - dur

    def get_end_time(self, solution: SchedulingSolution) -> int | AbsentValue:
        if not solution.is_present(self.task):
            return AbsentValue.ABSENT
        match self.start_or_end:
            case StartOrEnd.START:
                dur = solution.get_duration(self.task)
                return solution.get_start_time(self.task) * self.multiply_factor + dur
            case StartOrEnd.END:
                return solution.get_end_time(self.task) * self.multiply_factor

    def is_active(self, solution: SchedulingSolution) -> bool:
        return solution.is_present(self.task)

    def get_tasks(self) -> frozenset[Task]:
        return frozenset({self.task})

    @property
    def entity_id(self) -> Hashable:
        return "mult", self.task, self.multiply_factor, self.start_or_end


@dataclass(frozen=True)
class ModuloTaskEntity(SchedulingEntity[Task]):
    task: Task
    modulo_value: int

    def __post_init__(self):
        assert self.modulo_value > 0

    def get_start_time(self, solution: SchedulingSolution) -> int | AbsentValue:
        return solution.get_start_time(self.task) % self.modulo_value

    def get_end_time(self, solution: SchedulingSolution) -> int | AbsentValue:
        return solution.get_end_time(self.task) % self.modulo_value

    def is_active(self, solution: SchedulingSolution) -> bool:
        return solution.is_present(self.task)

    def get_tasks(self) -> frozenset[Task]:
        return frozenset({self.task})

    @property
    def entity_id(self) -> Hashable:
        return "task-modulo", self.task, self.modulo_value


@dataclass(frozen=True)
class VariableBatchEntity(SchedulingEntity[Task]):
    """Entity representing a batch with variable task membership.

    Unlike GroupEntity where tasks are fixed, VariableBatchEntity's membership
    is determined by the solution. This is essential for batching problems where:
    - The problem defines candidate tasks and batch IDs
    - The solution assigns tasks to batches
    - Constraints apply to batches (size limits, time windows, resource blocking)

    The entity queries the solution (which must be a BatchingSolution) to determine
    which tasks belong to this batch.

    Attributes:
        batch_id: Unique identifier for this batch
        candidate_tasks: Set of tasks that COULD be in this batch (used for get_tasks())

    Examples:
        >>> from discrete_optimization.generic_tasks_tools.batching import (
        ...     BatchingProblem, BatchingSolution
        ... )
        >>> # Define batch entities in problem
        >>> batch1 = VariableBatchEntity("batch_1", frozenset({0, 1, 2, 3, 4}))
        >>> batch2 = VariableBatchEntity("batch_2", frozenset({0, 1, 2, 3, 4}))
        >>>
        >>> # Solution must inherit from BatchingSolution and implement:
        >>> class MySolution(SchedulingSolution, BatchingSolution):
        ...     def get_task_batch_assignment(self, task):
        ...         return self.job_to_batch[task]  # Return batch ID for task
        >>>
        >>> # Query batch1 span
        >>> batch1.get_start_time(solution)  # min(start of jobs 0, 1, 4)
        >>> batch1.get_end_time(solution)    # max(end of jobs 0, 1, 4)
        >>> batch1.is_active(solution)       # True (has assigned tasks)

    Requires:
        Solution must inherit from BatchingSolution mixin (batching.py)
        which provides get_batch_membership() and get_task_batch_assignment()
    """

    batch_id: Hashable
    candidate_tasks: frozenset[Task] = frozenset()

    def _get_assigned_tasks(self, solution: BatchingSolution[Task]) -> frozenset[Task]:
        """Get tasks actually assigned to this batch in the solution.

        Args:
            solution: Must be a BatchingSolution instance

        Returns:
            Frozenset of tasks in this batch

        Raises:
            AttributeError: If solution doesn't have get_batch_membership method
        """
        # Solution must implement BatchingSolution mixin
        # Use get_batch_membership which is guaranteed by the mixin
        return frozenset(solution.get_batch_membership(self.batch_id))

    def get_start_time(self, solution: SchedulingSolution) -> int | AbsentValue:
        assigned_tasks = self._get_assigned_tasks(solution)
        if not assigned_tasks:
            return AbsentValue.ABSENT

        # Start time = earliest start of tasks in batch
        active_tasks = [task for task in assigned_tasks if solution.is_present(task)]
        if not active_tasks:
            return AbsentValue.ABSENT

        return min(solution.get_start_time(task) for task in active_tasks)

    def get_end_time(self, solution: SchedulingSolution) -> int | AbsentValue:
        assigned_tasks = self._get_assigned_tasks(solution)
        if not assigned_tasks:
            return AbsentValue.ABSENT

        # End time = latest end of tasks in batch
        active_tasks = [task for task in assigned_tasks if solution.is_present(task)]
        if not active_tasks:
            return AbsentValue.ABSENT

        return max(solution.get_end_time(task) for task in active_tasks)

    def is_active(self, solution: SchedulingSolution) -> bool:
        """Batch is active if it has at least one assigned task."""
        try:
            assigned_tasks = self._get_assigned_tasks(solution)
            return len(assigned_tasks) > 0 and any(
                solution.is_present(task) for task in assigned_tasks
            )
        except AttributeError:
            # Solution doesn't implement BatchingSolution
            return False

    def get_tasks(self) -> frozenset[Task]:
        """Return candidate tasks (actual assignment depends on solution)."""
        return self.candidate_tasks

    @property
    def entity_id(self) -> Hashable:
        return ("variable_batch", self.batch_id)


@dataclass(frozen=True)
class DummyEntity(SchedulingEntity[Task]):
    start: int
    end: int
    attribute: int  # to compute potential setupcost/time
    id_entity: Hashable

    def get_tasks(self) -> frozenset[Task]:
        return frozenset()

    def is_active(self, solution: SchedulingSolution) -> bool:
        return True

    def get_start_time(self, solution: SchedulingSolution) -> int | AbsentValue:
        return self.start

    def get_end_time(self, solution: SchedulingSolution) -> int | AbsentValue:
        return self.end

    def entity_id(self) -> Hashable:
        return self.id_entity


@dataclass(frozen=True)
class ModuloTwoEntity(SchedulingEntity[Task]):
    entity: SchedulingEntity[Task]
    modulo_entity: SchedulingEntity[Task]
    modulo_value_start_or_end: StartOrEnd = StartOrEnd.START

    @optional_override_implem
    def unfold_entities(self):
        yield self.entity
        self.entity.unfold_entities()
        yield self.modulo_entity
        self.modulo_entity.unfold_entities()

    def get_tasks(self) -> frozenset[Task]:
        return self.entity.get_tasks()

    def is_active(self, solution: SchedulingSolution) -> bool:
        return self.entity.is_active(solution)

    def get_start_time(self, solution: SchedulingSolution) -> int | AbsentValue:
        modulo = self.get_modulo_value(solution)
        if modulo == AbsentValue.ABSENT:
            return self.entity.get_start_time(solution)
        else:
            st = self.entity.get_start_time(solution)
            if st == AbsentValue.ABSENT:
                return st
            else:
                return st % modulo

    def get_end_time(self, solution: SchedulingSolution) -> int | AbsentValue:
        if self.entity.is_active(solution):
            dur = self.entity.get_end_time(solution) - self.entity.get_start_time(
                solution
            )
            st = self.get_start_time(solution)
            if st != AbsentValue.ABSENT:
                return st + dur
        return AbsentValue.ABSENT

    def get_modulo_value(self, solution: SchedulingSolution) -> int | AbsentValue:
        modulo = None
        match self.modulo_value_start_or_end:
            case StartOrEnd.START:
                modulo = self.modulo_entity.get_start_time(solution)
            case StartOrEnd.END:
                modulo = self.modulo_entity.get_end_time(solution)
        return modulo

    @property
    def entity_id(self) -> Hashable:
        return (
            "entity-modulo",
            self.entity.entity_id,
            self.modulo_entity.entity_id,
            self.modulo_value_start_or_end,
        )
