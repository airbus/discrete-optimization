#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Module for batching problems.

Provides mixins for problems where tasks are grouped into batches:
- Fixed batching: Batch membership is predetermined (Batch dataclass)
- Variable batching: Batch membership is determined by the solution (VariableBatchEntity + BatchingSolution)

Variable batching is useful for:
- Single batch processing: Assign jobs to batches on a machine
- Multi-machine batching: Assign jobs to batches on multiple machines
- Oven scheduling: Group items into batches with compatibility constraints
"""

import logging
from collections.abc import Hashable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

from discrete_optimization.generic_tasks_tools.base import (
    Task,
    TasksProblem,
    TasksSolution,
)
from discrete_optimization.generic_tasks_tools.enums import UndefinedData
from discrete_optimization.generic_tasks_tools.utils import optional_override

if TYPE_CHECKING:
    from discrete_optimization.generic_tasks_tools.entities import VariableBatchEntity

logger = logging.getLogger(__name__)

BatchId = TypeVar("BatchId", bound=Hashable)


@dataclass(frozen=True)
class Batch(Generic[Task]):
    """Fixed batch with predetermined task membership.

    Used for problems where batching decisions are made upfront.
    """

    batch_id: Hashable
    tasks: frozenset[Task]
    max_size: int | None = None
    min_size: int | None = None


class BatchingProblem(TasksProblem[Task], Generic[Task]):
    """Mixin for problems with variable batch assignment.

    This mixin supports problems where:
    - Tasks can be assigned to different batches
    - Batch membership is a decision variable
    - Constraints apply to batches (size, time windows, resources)

    **Architecture**: Batch entities are the PRIMARY abstraction.
    Create VariableBatchEntity instances first, then pass them to the problem.

    Example:
        >>> from discrete_optimization.generic_tasks_tools.entities import VariableBatchEntity
        >>>
        >>> # 1. Create batch entities FIRST (before problem)
        >>> batch_entities = [
        ...     VariableBatchEntity(batch_id=0, candidate_tasks=frozenset(tasks)),
        ...     VariableBatchEntity(batch_id=1, candidate_tasks=frozenset(tasks)),
        ...     VariableBatchEntity(batch_id=2, candidate_tasks=frozenset(tasks)),
        ... ]
        >>>
        >>> # 2. Define problem using entities
        >>> class MyProblem(BatchingProblem, EntityConstraintsProblem):
        ...     def __init__(self, tasks, batch_entities):
        ...         self._tasks = tasks
        ...         self.batch_entities = batch_entities  # Store entities
        ...
        ...     def get_batch_entities(self):
        ...         return self.batch_entities  # Return stored entities
        ...
        ...     def get_entity_precedence_constraints(self):
        ...         # Use entities directly in constraints
        ...         return {
        ...             self.batch_entities[0]: [self.batch_entities[1]]
        ...         }
        >>>
        >>> # 3. Create problem with entities
        >>> problem = MyProblem(tasks, batch_entities)
    """

    @optional_override
    def get_batch_entities(self) -> list["VariableBatchEntity[Task]"]:
        """Get all batch entities in the problem.

        This is the PRIMARY method - batch entities are created first,
        before the problem, and stored. This method returns them.

        Returns:
            List of VariableBatchEntity instances

        Example:
            >>> def __init__(self, tasks, batch_entities):
            ...     self._tasks = tasks
            ...     self.batch_entities = batch_entities
            ...
            >>> def get_batch_entities(self):
            ...     return self.batch_entities
        """
        return []

    def get_batch_ids(self) -> list[Hashable]:
        """Get list of batch IDs (derived from entities).

        This is DERIVED from get_batch_entities().
        You don't override this - it extracts batch_id from each entity.

        Returns:
            List of batch IDs (batch_id attribute from each entity)
        """
        return [entity.batch_id for entity in self.get_batch_entities()]

    def get_batch_entity(self, batch_id: Hashable) -> "VariableBatchEntity[Task]":
        """Get batch entity for a specific batch_id.

        Args:
            batch_id: Batch identifier

        Returns:
            VariableBatchEntity with matching batch_id

        Raises:
            ValueError: If batch_id not found
        """
        for entity in self.get_batch_entities():
            if entity.batch_id == batch_id:
                return entity
        raise ValueError(f"Batch entity with batch_id={batch_id} not found")

    @optional_override
    def get_fixed_batches(self) -> list[Batch]:
        """Get batches with predetermined task membership.

        Returns:
            List of Batch objects with fixed task assignment.

        Default: No fixed batches.
        """
        return []

    def get_candidate_tasks_for_batch(self, batch_id: Hashable) -> frozenset[Task]:
        """Get tasks that could potentially be assigned to this batch.

        Args:
            batch_id: Batch identifier

        Returns:
            Set of tasks that are candidates for this batch

        Default: Extract from batch entity's candidate_tasks.
        """
        return self.get_batch_entity(batch_id).candidate_tasks

    @optional_override
    def get_batch_size_bounds(
        self, batch_id: Hashable
    ) -> tuple[int | None, int | None]:
        """Get minimum and maximum NUMBER OF TASKS in a batch.

        This is about cardinality (count), not weight/capacity.
        For capacity constraints, use get_task_size() and get_batch_capacity().

        Args:
            batch_id: Batch identifier

        Returns:
            Tuple of (min_size, max_size). None means no constraint.

        Default: No size constraints.
        """
        return None, None

    @optional_override
    def get_task_size(self, task: Task) -> int | float | tuple[int | float, ...]:
        """Get size/weight of a task for bin packing.

        Task size is used in batch capacity constraints:
        sum(task_sizes in batch) <= batch_capacity

        Size can be:
        - Scalar: single dimension (weight, volume, processing time)
        - Tuple: multiple dimensions (weight AND volume)

        Args:
            task: Task to query

        Returns:
            Task size (int, float, or tuple for multi-dimensional)

        Default: Size 1 for all tasks.

        Example:
            >>> # Oven scheduling: tasks have physical size
            >>> def get_task_size(self, task):
            ...     return self.tasks_data[task].size
            >>>
            >>> # Multi-dimensional: (weight, volume)
            >>> def get_task_size(self, task):
            ...     return (self.task_weight[task], self.task_volume[task])
        """
        return 1

    @optional_override
    def get_batch_capacity(
        self, batch_id: Hashable
    ) -> int | float | tuple[int | float, ...]:
        """Get capacity of a batch for bin packing.

        Batch capacity constrains total size of tasks in batch:
        sum(get_task_size(task) for task in batch) <= get_batch_capacity(batch)

        For multi-dimensional sizes (tuples), capacity is also a tuple and
        constraint applies element-wise.

        Args:
            batch_id: Batch identifier

        Returns:
            Batch capacity (int, float, or tuple for multi-dimensional)

        Default: Infinite capacity (no constraint).

        Example:
            >>> # Oven scheduling: machine has capacity
            >>> def get_batch_capacity(self, batch_id):
            ...     machine_id = self._get_machine_from_batch(batch_id)
            ...     return self.machines_data[machine_id].capacity
            >>>
            >>> # Multi-dimensional: (max_weight, max_volume)
            >>> def get_batch_capacity(self, batch_id):
            ...     return (self.max_weight, self.max_volume)
        """
        return float("inf")

    @optional_override
    def is_batch_optional(self, batch_id: Hashable) -> bool:
        """Check if batch can be empty (no tasks assigned).

        Args:
            batch_id: Batch identifier

        Returns:
            True if batch can be empty, False if it must have at least one task

        Default: All batches are optional.
        """
        return True

    @optional_override
    def get_task_attribute(self, task: Task) -> Hashable | None:
        """Get attribute of a task.

        Task attribute determines batching compatibility and batch characteristics.
        Common examples:
        - Product type/color/recipe in manufacturing
        - Priority class in scheduling
        - Resource requirements

        Args:
            task: Task to query

        Returns:
            Task attribute (any hashable type), or None if no attribute concept

        Default: No task attributes.

        Example:
            >>> # Oven scheduling: tasks have product types
            >>> def get_task_attribute(self, task):
            ...     return self.tasks_data[task].attribute  # 0, 1, 2, ...
        """
        return None

    @optional_override
    def can_batch_together(self, task1: Task, task2: Task) -> bool:
        """Check if two tasks are compatible for batching together.

        Tasks are incompatible if they cannot be in the same batch due to:
        - Different attributes (e.g., different product types)
        - Conflicting resource requirements
        - Incompatible processing characteristics

        Args:
            task1: First task
            task2: Second task

        Returns:
            True if tasks can be batched together, False otherwise

        Default: All tasks are compatible (can batch together).

        Example:
            >>> # Oven scheduling: only same-attribute tasks can batch
            >>> def can_batch_together(self, task1, task2):
            ...     return self.get_task_attribute(task1) == self.get_task_attribute(task2)
            >>>
            >>> # Size-based compatibility
            >>> def can_batch_together(self, task1, task2):
            ...     return self.task_sizes[task1] + self.task_sizes[task2] <= self.max_batch_size
        """
        return True

    @optional_override
    def get_batch_attribute(
        self, batch_id: Hashable, tasks: frozenset[Task]
    ) -> Hashable | None:
        """Determine batch attribute based on tasks assigned to it.

        Batch attribute is derived from tasks in the batch and affects:
        - Setup time when switching between batches
        - Processing time of the batch
        - Resource consumption

        Args:
            batch_id: Batch identifier
            tasks: Tasks assigned to this batch

        Returns:
            Batch attribute, or None if no attribute concept

        Default: If all tasks have same attribute, return it; else None.

        Example:
            >>> # Oven: batch inherits common attribute
            >>> def get_batch_attribute(self, batch_id, tasks):
            ...     if not tasks:
            ...         return None
            ...     attrs = {self.get_task_attribute(t) for t in tasks}
            ...     return attrs.pop() if len(attrs) == 1 else None  # All must match
        """
        if not tasks:
            return None

        # Default: if all tasks have same attribute, batch gets that attribute
        task_attrs = {self.get_task_attribute(t) for t in tasks}
        if None in task_attrs:
            return None  # No attribute concept
        if len(task_attrs) == 1:
            return task_attrs.pop()  # All same
        return None  # Mixed attributes - invalid batch

    @optional_override
    def get_batch_duration_bounds(
        self, batch_id: Hashable, tasks: frozenset[Task]
    ) -> tuple[int | None, int | None]:
        """Get minimum and maximum duration for a batch based on its tasks.

        Batch duration depends on tasks assigned to it and batch attribute.
        This is separate from size bounds (number of tasks).

        Args:
            batch_id: Batch identifier
            tasks: Tasks assigned to this batch

        Returns:
            Tuple of (min_duration, max_duration). None means no constraint.

        Default: No duration constraints.

        Example:
            >>> # Oven: duration based on task min/max durations
            >>> def get_batch_duration_bounds(self, batch_id, tasks):
            ...     if not tasks:
            ...         return None, None
            ...     min_dur = max(self.tasks_data[t].min_duration for t in tasks)
            ...     max_dur = min(self.tasks_data[t].max_duration for t in tasks)
            ...     return min_dur, max_dur
        """
        return None, None

    @optional_override
    def get_min_task_duration_in_batch(self, task: Task) -> int:
        return UndefinedData.UNDEFINED

    @optional_override
    def get_max_task_duration_in_batch(self, task: Task) -> int:
        return UndefinedData.UNDEFINED

    @optional_override
    def must_be_batched(self, task: Task) -> bool:
        """Check if a task MUST be assigned to a batch.

        Args:
            task: Task to query

        Returns:
            True if task must be in a batch, False if batching is optional

        Default: Batching is optional for all tasks.

        Example:
            >>> # Some products must be batched, others can run individually
            >>> def must_be_batched(self, task):
            ...     return self.tasks_data[task].requires_batching
        """
        return False

    @optional_override
    def can_be_batched(self, task: Task) -> bool:
        """Check if a task CAN be assigned to a batch.

        Args:
            task: Task to query

        Returns:
            True if task can be in a batch, False if it must run individually

        Default: All tasks can be batched.

        Example:
            >>> # Large items cannot be batched
            >>> def can_be_batched(self, task):
            ...     return self.get_task_size(task) <= self.single_task_threshold
        """
        return True

    def get_batching_task_categories(self) -> dict[str, list[Task]]:
        """Categorize tasks based on batching requirements.

        Returns:
            Dictionary with three keys:
            - 'mandatory': Tasks that MUST be batched (must_be_batched() = True)
            - 'optional': Tasks that CAN be batched but don't have to (can_be_batched() = True, must_be_batched() = False)
            - 'forbidden': Tasks that CANNOT be batched (can_be_batched() = False)

        Note: Tasks with must_be_batched() = True and can_be_batched() = False are invalid
        and will raise a ValueError.

        Example:
            >>> categories = problem.get_batching_task_categories()
            >>> print(f"Must batch: {categories['mandatory']}")
            >>> print(f"Can batch: {categories['optional']}")
            >>> print(f"Cannot batch: {categories['forbidden']}")
        """
        mandatory_batch = []
        optional_batch = []
        forbidden_batch = []

        for task in self.tasks_list:
            must_batch = self.must_be_batched(task)
            can_batch = self.can_be_batched(task)

            if must_batch and not can_batch:
                raise ValueError(
                    f"Task {task} has contradictory batching requirements: "
                    f"must_be_batched=True but can_be_batched=False"
                )

            if must_batch:
                mandatory_batch.append(task)
            elif can_batch:
                optional_batch.append(task)
            else:
                forbidden_batch.append(task)

        return {
            "mandatory": mandatory_batch,
            "optional": optional_batch,
            "forbidden": forbidden_batch,
        }

    @optional_override
    def get_batch_presence_chains(self) -> list[list[Hashable]]:
        """Get batch presence ordering chains for symmetry breaking.

        Returns chains of batches where if batch[i+1] is used, batch[i] must be used.
        This helps avoid symmetric solutions by enforcing a canonical ordering.

        Returns:
            List of batch chains. Each chain is a list of batch IDs in order.
            Empty list means no presence constraints (all batches independent).

        Example - Single machine (one chain):
            >>> def get_batch_presence_chains(self):
            ...     return [self.get_batch_ids()]  # All batches in one chain

        Example - Multi-machine (multiple chains):
            >>> def get_batch_presence_chains(self):
            ...     # Batches 0-4 for machine 1, batches 5-9 for machine 2
            ...     return [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]

        Example - No presence constraints:
            >>> def get_batch_presence_chains(self):
            ...     return []  # Batches can be used in any order
        """
        return []


class BatchingSolution(TasksSolution[Task], Generic[Task]):
    """Mixin for solutions to batching problems.

    Provides abstract methods for querying batch assignments.
    Concrete solutions must implement at least one of:
    - get_batch_membership(batch_id) -> set[Task]
    - get_task_batch_assignment(task) -> BatchId

    These methods are used by VariableBatchEntity to determine batch spans.
    """

    problem: BatchingProblem[Task]

    @optional_override
    def get_task_batch_assignment(self, task: Task) -> Hashable | None:
        """Get which batch a task is assigned to.

        Args:
            task: Task to query

        Returns:
            Batch ID that task is assigned to, or None if unassigned
        """
        return None

    def get_batch_membership(self, batch_id: Hashable) -> set[Task]:
        """Get all tasks assigned to a batch.

        Default implementation queries each task. Can be overridden for efficiency.

        Args:
            batch_id: Batch to query

        Returns:
            Set of tasks in this batch
        """
        return {
            task
            for task in self.problem.tasks_list
            if self.get_task_batch_assignment(task) == batch_id
        }

    def get_batch_size(self, batch_id: Hashable) -> int:
        """Get number of tasks in a batch.

        Args:
            batch_id: Batch to query

        Returns:
            Number of tasks assigned to this batch
        """
        return len(self.get_batch_membership(batch_id))

    def check_batch_size_constraints(self) -> bool:
        """Check that all batches satisfy size constraints.

        Returns:
            True if all batch sizes are within bounds, False otherwise
        """
        for batch_id in self.problem.get_batch_ids():
            size = self.get_batch_size(batch_id)
            min_size, max_size = self.problem.get_batch_size_bounds(batch_id)

            # Check minimum size
            if min_size is not None:
                # Empty batches allowed if batch is optional
                if size > 0 or not self.problem.is_batch_optional(batch_id):
                    if size < min_size:
                        logger.debug(
                            f"Batch {batch_id} has {size} tasks, less than minimum {min_size}"
                        )
                        return False

            # Check maximum size
            if max_size is not None and size > max_size:
                logger.debug(
                    f"Batch {batch_id} has {size} tasks, exceeds maximum {max_size}"
                )
                return False

        return True

    def check_task_batch_uniqueness(self) -> bool:
        """Check that each task is assigned to at most one batch.

        Returns:
            True if no task appears in multiple batches, False otherwise
        """
        task_to_batches = {}
        for batch_id in self.problem.get_batch_ids():
            for task in self.get_batch_membership(batch_id):
                if task in task_to_batches:
                    logger.debug(
                        f"Task {task} appears in multiple batches: "
                        f"{task_to_batches[task]} and {batch_id}"
                    )
                    return False
                task_to_batches[task] = batch_id

        return True

    def check_batch_compatibility(self) -> bool:
        """Check that all tasks in each batch are compatible.

        Returns:
            True if all batches contain only compatible tasks, False otherwise
        """
        for batch_id in self.problem.get_batch_ids():
            tasks = list(self.get_batch_membership(batch_id))
            if len(tasks) < 2:
                continue  # Single task or empty batch, always compatible

            # Check all pairs
            for i in range(len(tasks)):
                for j in range(i + 1, len(tasks)):
                    if not self.problem.can_batch_together(tasks[i], tasks[j]):
                        logger.debug(
                            f"Incompatible tasks {tasks[i]} and {tasks[j]} "
                            f"in batch {batch_id}"
                        )
                        return False

        return True

    def check_batch_attributes(self) -> bool:
        """Check that each batch has a valid attribute.

        For batches with task attribute concept, ensures batch can be assigned
        a valid attribute based on its tasks.

        Returns:
            True if all batches have valid attributes, False otherwise
        """
        for batch_id in self.problem.get_batch_ids():
            tasks = frozenset(self.get_batch_membership(batch_id))
            if not tasks:
                continue  # Empty batch, no attribute needed

            batch_attr = self.problem.get_batch_attribute(batch_id, tasks)
            if (
                batch_attr is None
                and self.problem.get_task_attribute(next(iter(tasks))) is not None
            ):
                # Problem uses attributes, but batch has invalid attribute
                logger.debug(
                    f"Batch {batch_id} with tasks {tasks} has no valid attribute"
                )
                return False

        return True

    def check_batch_capacity(self) -> bool:
        """Check that all batches respect capacity constraints.

        Validates: sum(task_sizes) <= batch_capacity for each batch.
        Supports scalar and multi-dimensional sizes/capacities.

        Returns:
            True if all batches within capacity, False otherwise
        """
        for batch_id in self.problem.get_batch_ids():
            tasks = list(self.get_batch_membership(batch_id))
            if not tasks:
                continue  # Empty batch, no capacity constraint

            # Get batch capacity
            capacity = self.problem.get_batch_capacity(batch_id)
            if capacity == float("inf"):
                continue  # No capacity constraint

            # Get task sizes
            task_sizes = [self.problem.get_task_size(task) for task in tasks]

            # Check if multi-dimensional
            if isinstance(task_sizes[0], tuple):
                # Multi-dimensional: check each dimension
                total_size = tuple(
                    sum(sizes[i] for sizes in task_sizes)
                    for i in range(len(task_sizes[0]))
                )
                if not isinstance(capacity, tuple):
                    logger.debug(
                        f"Multi-dimensional task sizes but scalar capacity for batch {batch_id}"
                    )
                    return False

                for dim, (total, cap) in enumerate(zip(total_size, capacity)):
                    if total > cap:
                        logger.debug(
                            f"Batch {batch_id} dimension {dim} exceeds capacity: "
                            f"{total} > {cap}"
                        )
                        return False
            else:
                # Scalar: simple sum
                total_size = sum(task_sizes)
                if total_size > capacity:
                    logger.debug(
                        f"Batch {batch_id} exceeds capacity: {total_size} > {capacity}"
                    )
                    return False

        return True
