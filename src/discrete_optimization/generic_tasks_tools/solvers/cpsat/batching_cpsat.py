#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""CP-SAT solver for batching problems.

Handles task assignment to batches with:
- Batch size constraints (min/max number of tasks)
- Task compatibility (which tasks can batch together)
- Task attributes (batches inherit from tasks)
- Batch duration bounds (min/max duration based on tasks)

Designed for problems like:
- Single batch processing machine
- Oven scheduling with product types
- Multi-machine batching with compatibility
"""

from __future__ import annotations

import logging
from collections.abc import Hashable
from typing import Generic

from ortools.sat.python import cp_model

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.generic_scheduling import (
    GenericSchedulingProblem,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.scheduling import (
    SchedulingCpSatSolver,
)

logger = logging.getLogger(__name__)


class BatchingCPSatSolver(
    SchedulingCpSatSolver[Task],
    Generic[Task],
):
    """CP-SAT solver mixin for batching problems.

    Creates decision variables for:
    - Task-to-batch assignment
    - Batch presence (is batch used?)
    - Batch attributes (if applicable)
    - Batch start/end times
    - Setup times between consecutive batches

    Enforces constraints:
    - Each task in exactly one batch
    - Batch size bounds
    - Task compatibility within batches
    - Batch attribute consistency
    - Batch duration bounds
    """

    problem: GenericSchedulingProblem

    # Decision variables (created in init_model)
    task_to_batch_vars: dict[Task, dict[Hashable, cp_model.IntVar]]
    batch_used_vars: dict[Hashable, cp_model.IntVar]
    batch_attr_vars: dict[Hashable, cp_model.IntVar | None]
    batch_start_vars: dict[Hashable, cp_model.IntVar]
    batch_end_vars: dict[Hashable, cp_model.IntVar]
    batch_size_vars: dict[Hashable, cp_model.IntVar]

    def init_batching_variables_and_constraints(self) -> None:
        """Initialize batching decision variables and constraints.

        Should be called from the concrete solver's init_model() method.
        """
        logger.info("Initializing batching variables and constraints...")

        # Initialize decision variable dictionaries
        self.task_to_batch_vars = {}
        self.batch_used_vars = {}
        self.batch_attr_vars = {}
        self.batch_size_vars = {}
        batch_entities = self.problem.get_batch_entities()

        # Get problem data
        tasks = self.problem.tasks_list
        batches = self.problem.get_batch_ids()
        horizon = self.problem.get_makespan_upper_bound()
        # 1. Create task-to-batch assignment variables
        self._create_assignment_variables(tasks, batches)
        # 2. Create batch variables (used, attribute, start, end, size)
        self._create_batch_variables(batches, horizon)
        # 3. Add assignment constraints
        self._add_assignment_constraints(tasks)
        # 4. Add batch size constraints
        self._add_batch_size_constraints(tasks, batches)
        # 5. Add compatibility constraints
        self._add_compatibility_constraints(tasks, batches)
        # 6. Add attribute constraints (if applicable)
        self._add_attribute_constraints(tasks, batches)

        # 8. Add batch presence chain constraints (symmetry breaking)
        self._add_batch_presence_chain_constraints()

        logger.info("Batching variables and constraints initialized successfully.")

    def _create_assignment_variables(
        self, tasks: list[Task], batches: list[Hashable]
    ) -> None:
        """Create binary variables for task-to-batch assignment.

        task_to_batch[task][batch] = 1 if task assigned to batch, 0 otherwise

        Only creates variables for tasks that can_be_batched().
        Tasks that cannot be batched are excluded from batch assignment.
        """
        for task in tasks:
            # Skip tasks that cannot be batched
            if not self.problem.can_be_batched(task):
                continue

            self.task_to_batch_vars[task] = {}
            candidates = self.problem.get_candidate_tasks_for_batch

            for batch in batches:
                # Only create variable if task is candidate for this batch
                if task in candidates(batch):
                    var_name = f"task_{task}_to_batch_{batch}"
                    self.task_to_batch_vars[task][batch] = self.cp_model.NewBoolVar(
                        var_name
                    )

    def _create_batch_variables(self, batches: list[Hashable], horizon: int) -> None:
        """Create variables for batch properties."""
        # Determine if problem uses task attributes
        uses_attributes = any(
            self.problem.get_task_attribute(t) is not None
            for t in self.problem.tasks_list
        )

        if uses_attributes:
            # Collect all possible attribute values
            all_attributes = {
                self.problem.get_task_attribute(t)
                for t in self.problem.tasks_list
                if self.problem.get_task_attribute(t) is not None
            }
            # Convert to integers for CP-SAT
            attr_to_int = {
                attr: i for i, attr in enumerate(sorted(all_attributes, key=str))
            }
            self.attr_to_int = attr_to_int
            self.int_to_attr = {i: attr for attr, i in attr_to_int.items()}

        for batch in batches:
            # Batch used (presence)
            self.batch_used_vars[batch] = self.cp_model.NewBoolVar(
                f"batch_{batch}_used"
            )

            # Batch attribute (if applicable)
            if uses_attributes:
                max_attr_int = max(self.attr_to_int.values())
                self.batch_attr_vars[batch] = self.cp_model.NewIntVar(
                    0, max_attr_int, f"batch_{batch}_attr"
                )
            else:
                self.batch_attr_vars[batch] = None
            # Batch size (number of tasks)
            max_size = len(self.problem.tasks_list)
            self.batch_size_vars[batch] = self.cp_model.NewIntVar(
                0, max_size, f"batch_{batch}_size"
            )

    def _add_assignment_constraints(self, tasks: list[Task]) -> None:
        """Add constraints on task-to-batch assignment.

        - Tasks that must_be_batched(): exactly one batch
        - Tasks that can_be_batched() but not mandatory: at most one batch
        - Tasks that cannot be batched: no assignment variables (already filtered)
        """
        for task in tasks:
            if task not in self.task_to_batch_vars:
                # Task cannot be batched, skip
                continue

            assignment_vars = list(self.task_to_batch_vars[task].values())
            if not assignment_vars:
                # No candidate batches for this task
                if self.problem.must_be_batched(task):
                    logger.error(
                        f"Task {task} must be batched but has no candidate batches!"
                    )
                continue

            if self.problem.must_be_batched(task):
                # Task MUST be in exactly one batch
                self.cp_model.Add(sum(assignment_vars) == 1)
            else:
                # Task CAN be batched: at most one batch
                self.cp_model.Add(sum(assignment_vars) <= 1)

    def _add_batch_size_constraints(
        self, tasks: list[Task], batches: list[Hashable]
    ) -> None:
        """Add batch size constraints (cardinality and capacity)."""
        for batch in batches:
            # Batch size = number of tasks assigned to it
            assigned_vars = [
                self.task_to_batch_vars[task].get(batch, 0)
                for task in tasks
                if batch in self.task_to_batch_vars[task]
            ]

            if assigned_vars:
                # Link size variable to assignment
                self.cp_model.Add(self.batch_size_vars[batch] == sum(assigned_vars))

                # Batch used iff size > 0
                self.cp_model.Add(self.batch_size_vars[batch] > 0).OnlyEnforceIf(
                    self.batch_used_vars[batch]
                )
                self.cp_model.Add(self.batch_size_vars[batch] == 0).OnlyEnforceIf(
                    self.batch_used_vars[batch].Not()
                )

                # Min/max cardinality bounds (number of tasks)
                min_size, max_size = self.problem.get_batch_size_bounds(batch)

                if min_size is not None:
                    # If batch is used and not optional, must meet min size
                    if not self.problem.is_batch_optional(batch):
                        self.cp_model.Add(self.batch_size_vars[batch] >= min_size)
                    else:
                        # If used, meet min size
                        self.cp_model.Add(
                            self.batch_size_vars[batch] >= min_size
                        ).OnlyEnforceIf(self.batch_used_vars[batch])

                if max_size is not None:
                    self.cp_model.Add(self.batch_size_vars[batch] <= max_size)

                # Capacity constraints (bin packing)
                self._add_capacity_constraint_for_batch(batch, tasks)

    def _add_compatibility_constraints(
        self, tasks: list[Task], batches: list[Hashable]
    ) -> None:
        """Add constraints preventing incompatible tasks in same batch."""
        for batch in batches:
            # For each pair of tasks, if both assigned to batch, must be compatible
            tasks_for_batch = [
                task for task in tasks if batch in self.task_to_batch_vars[task]
            ]

            for i, task1 in enumerate(tasks_for_batch):
                for task2 in tasks_for_batch[i + 1 :]:
                    if not self.problem.can_batch_together(task1, task2):
                        # If task1 in batch, task2 cannot be in batch
                        var1 = self.task_to_batch_vars[task1][batch]
                        var2 = self.task_to_batch_vars[task2][batch]
                        self.cp_model.Add(var1 + var2 <= 1)

    def _add_attribute_constraints(
        self, tasks: list[Task], batches: list[Hashable]
    ) -> None:
        """Add constraints on batch attributes."""

        for batch in batches:
            batch_attr_var = self.batch_attr_vars[batch]
            if batch_attr_var is None:
                continue

            # For each task, if assigned to batch, batch must have task's attribute
            for task in tasks:
                if batch not in self.task_to_batch_vars[task]:
                    continue

                task_attr = self.problem.get_task_attribute(task)
                if task_attr is None:
                    continue

                task_attr_int = self.attr_to_int[task_attr]
                assignment_var = self.task_to_batch_vars[task][batch]

                # If task assigned to batch, batch attr = task attr
                self.cp_model.Add(batch_attr_var == task_attr_int).OnlyEnforceIf(
                    assignment_var
                )

    def _add_capacity_constraint_for_batch(
        self, batch: Hashable, tasks: list[Task]
    ) -> None:
        """Add bin-packing capacity constraint for a batch.

        Ensures: sum(task_sizes in batch) <= batch_capacity
        """
        capacity = self.problem.get_batch_capacity(batch)
        if capacity == float("inf"):
            return  # No capacity constraint

        # Get all tasks that could be in this batch
        candidate_tasks = [
            task for task in tasks if batch in self.task_to_batch_vars[task]
        ]

        if not candidate_tasks:
            return

        # Check if sizes are scalar or multi-dimensional
        sample_size = self.problem.get_task_size(candidate_tasks[0])

        if isinstance(sample_size, tuple):
            # Multi-dimensional capacity
            if not isinstance(capacity, tuple):
                logger.warning(
                    f"Multi-dimensional task sizes but scalar capacity for batch {batch}"
                )
                return

            num_dims = len(sample_size)
            # For each dimension
            for dim in range(num_dims):
                dim_capacity = capacity[dim]
                if dim_capacity == float("inf"):
                    continue

                # Convert to integer for CP-SAT
                dim_capacity_int = int(dim_capacity)

                # sum(task_size[dim] * assigned) <= capacity[dim]
                size_terms = []
                for task in candidate_tasks:
                    task_size = self.problem.get_task_size(task)
                    task_size_dim = int(task_size[dim])
                    if task_size_dim > 0:
                        assignment_var = self.task_to_batch_vars[task][batch]
                        size_terms.append(task_size_dim * assignment_var)

                if size_terms:
                    self.cp_model.Add(sum(size_terms) <= dim_capacity_int)

        else:
            # Scalar capacity
            if not isinstance(capacity, (int, float)):
                logger.warning(
                    f"Scalar task sizes but non-scalar capacity for batch {batch}"
                )
                return

            capacity_int = int(capacity)

            # sum(task_size * assigned) <= capacity
            size_terms = []
            for task in candidate_tasks:
                task_size = int(self.problem.get_task_size(task))
                if task_size > 0:
                    assignment_var = self.task_to_batch_vars[task][batch]
                    size_terms.append(task_size * assignment_var)

            if size_terms:
                self.cp_model.Add(sum(size_terms) <= capacity_int)

    def _add_batch_presence_chain_constraints(self) -> None:
        """Add batch presence chain constraints for symmetry breaking.

        Uses problem.get_batch_presence_chains() to get chains of batches
        where if batch[i+1] is used, batch[i] must be used.

        This avoids symmetric solutions by enforcing canonical batch ordering.
        """
        chains = self.problem.get_batch_presence_chains()

        for chain in chains:
            # For each chain, enforce: batch_used[i] >= batch_used[i+1]
            for i in range(len(chain) - 1):
                batch_i = chain[i]
                batch_j = chain[i + 1]

                # If batch j is used, batch i must be used
                if batch_i in self.batch_used_vars and batch_j in self.batch_used_vars:
                    self.cp_model.Add(
                        self.batch_used_vars[batch_i] >= self.batch_used_vars[batch_j]
                    )
