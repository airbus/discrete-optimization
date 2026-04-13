#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""
Vector-based solution encoding for the Oven Scheduling Problem.

This module provides a permutation-based encoding suitable for metaheuristic
algorithms like simulated annealing, hill climbing, and genetic algorithms.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from discrete_optimization.generic_tools.do_problem import Solution

if TYPE_CHECKING:
    from discrete_optimization.ovensched.problem import (
        Machine,
        OvenSchedulingProblem,
        ScheduleInfo,
    )

logger = logging.getLogger(__name__)


def decode_permutation_to_schedule(
    problem: "OvenSchedulingProblem",
    permutation: npt.NDArray[np.int_],
    max_open_batches_per_machine: int = 3,
) -> dict["Machine", list["ScheduleInfo"]]:
    """
    Decode a task permutation into a schedule using an improved greedy strategy.

    This decoder maintains multiple open batches per machine simultaneously,
    allowing better batching of tasks with the same attribute even if they
    appear at different positions in the permutation.

    Strategy:
        1. Process tasks in permutation order
        2. For each task, try to add it to any compatible open batch on eligible machines
        3. Compatible = same attribute, fits capacity, duration compatible
        4. If no compatible batch, start new one (closing oldest if max batches reached)
        5. Uses heap to manage batch priority (oldest batches get closed first)

    Args:
        problem: The OvenSchedulingProblem instance
        permutation: Array of task indices in scheduling order
        max_open_batches_per_machine: Maximum number of open batches per machine (default: 3)

    Returns:
        Dictionary mapping each machine to its list of scheduled batches
    """
    import heapq

    schedule_per_machine: dict[Machine, list[ScheduleInfo]] = {
        m: [] for m in range(problem.n_machines)
    }

    # Track multiple open batches per machine
    # Structure: {machine: [(priority, batch_id, batch_dict), ...]} as a min-heap
    # Priority is based on creation order (older batches have lower priority)
    open_batches: dict[Machine, list] = {m: [] for m in range(problem.n_machines)}
    batch_id_counter = 0

    for task_id in permutation:
        task_data = problem.tasks_data[task_id]
        eligible_machines = list(task_data.eligible_machines)

        if not eligible_machines:
            logger.warning(f"Task {task_id} has no eligible machines, skipping")
            continue

        # Try to find a compatible open batch on any eligible machine
        best_machine = None
        best_batch_idx = None
        batch_found = False

        for machine in eligible_machines:
            if not open_batches[machine]:
                continue

            # Check all open batches on this machine
            for idx, (priority, batch_id, batch) in enumerate(open_batches[machine]):
                # Check attribute compatibility
                if batch["attribute"] != task_data.attribute:
                    continue

                # Check capacity
                if (
                    batch["size"] + task_data.size
                    > problem.machines_data[machine].capacity
                ):
                    continue

                # Check duration compatibility
                new_min_dur = max(batch["min_dur"], task_data.min_duration)
                new_max_dur = min(batch["max_dur"], task_data.max_duration)

                if new_min_dur <= new_max_dur:
                    best_machine = machine
                    best_batch_idx = idx
                    batch_found = True
                    break

            if batch_found:
                break

        if batch_found:
            # Add task to existing batch
            priority, batch_id, batch = open_batches[best_machine][best_batch_idx]
            batch["tasks"].add(task_id)
            batch["size"] += task_data.size
            batch["min_dur"] = max(batch["min_dur"], task_data.min_duration)
            batch["max_dur"] = min(batch["max_dur"], task_data.max_duration)
            batch["last_access"] = task_id  # Track last access for priority
        else:
            # No compatible batch found, create new one
            # Choose best machine (first eligible for now, could be improved)
            best_machine = eligible_machines[0]

            # If max batches reached, close the oldest batch
            if len(open_batches[best_machine]) >= max_open_batches_per_machine:
                _close_oldest_batch(
                    problem, best_machine, open_batches, schedule_per_machine
                )

            # Create new batch
            new_batch = {
                "tasks": {task_id},
                "attribute": task_data.attribute,
                "size": task_data.size,
                "min_dur": task_data.min_duration,
                "max_dur": task_data.max_duration,
                "last_access": task_id,
            }

            # Add to heap (priority = creation order, lower is older)
            heapq.heappush(
                open_batches[best_machine],
                (batch_id_counter, batch_id_counter, new_batch),
            )
            batch_id_counter += 1

    # Close all remaining open batches
    for machine in range(problem.n_machines):
        while open_batches[machine]:
            _close_oldest_batch(problem, machine, open_batches, schedule_per_machine)

    return schedule_per_machine


def _close_oldest_batch(
    problem: "OvenSchedulingProblem",
    machine: "Machine",
    open_batches: dict["Machine", list],
    schedule_per_machine: dict["Machine", list["ScheduleInfo"]],
) -> None:
    """
    Close the oldest open batch on a machine (pop from heap).

    This function:
    - Removes the oldest batch from the heap
    - Computes batch timing considering setup times and availability
    - Adds the batch to the schedule

    Args:
        problem: The problem instance
        machine: Machine index
        open_batches: Heap of open batches per machine
        schedule_per_machine: Schedule being built (modified in place)
    """
    import heapq

    from discrete_optimization.ovensched.problem import ScheduleInfo

    if not open_batches[machine]:
        return

    # Pop oldest batch from heap
    priority, batch_id, batch_info = heapq.heappop(open_batches[machine])

    # Choose batch duration (minimum duration to maximize flexibility)
    batch_duration = batch_info["min_dur"]

    # Compute earliest start time considering previous batches
    candidate_start = 0

    if schedule_per_machine[machine]:
        # Not first batch: start after previous batch + setup
        last_batch = schedule_per_machine[machine][-1]
        candidate_start = last_batch.end_time
        prev_attr = last_batch.task_attribute
        setup_time = problem.setup_times[prev_attr][batch_info["attribute"]]
        candidate_start += setup_time
    else:
        # First batch: consider initial attribute setup
        initial_attr = problem.machines_data[machine].initial_attribute
        setup_time = problem.setup_times[initial_attr][batch_info["attribute"]]
        candidate_start += setup_time

    # CRITICAL: Respect earliest_start constraint of tasks in batch
    tasks_earliest_start = max(
        problem.tasks_data[t].earliest_start for t in batch_info["tasks"]
    )
    candidate_start = max(candidate_start, tasks_earliest_start)

    # Find valid availability window
    availability_windows = problem.machines_data[machine].availability
    start_time = None
    end_time = None

    for window_start, window_end in availability_windows:
        # Check if batch can fit in this window
        if window_end - window_start < batch_duration:
            continue

        # Try to place batch at candidate_start if it's in this window
        if (
            candidate_start >= window_start
            and candidate_start + batch_duration <= window_end
        ):
            start_time = candidate_start
            end_time = candidate_start + batch_duration
            break

        # Otherwise, try to place at start of window if window is after candidate
        if window_start >= candidate_start:
            if window_start + batch_duration <= window_end:
                start_time = window_start
                end_time = window_start + batch_duration
                break

    # If no valid window found, place anyway (solution will be infeasible but decoder completes)
    if start_time is None:
        start_time = candidate_start
        end_time = candidate_start + batch_duration

    # Create and add batch
    batch_idx = len(schedule_per_machine[machine])
    schedule_per_machine[machine].append(
        ScheduleInfo(
            tasks=batch_info["tasks"],
            task_attribute=batch_info["attribute"],
            start_time=start_time,
            end_time=end_time,
            machine_batch_index=(machine, batch_idx),
        )
    )


def extract_permutation_from_schedule(
    problem: "OvenSchedulingProblem",
    schedule_per_machine: dict["Machine", list["ScheduleInfo"]],
) -> npt.NDArray[np.int_]:
    """
    Extract a task permutation from a schedule.

    Tasks are sorted by their start time (earlier tasks first).
    If two tasks start at the same time, they are sorted by task ID.

    Args:
        problem: The problem instance (for validation)
        schedule_per_machine: The schedule to extract from

    Returns:
        Array of task indices in start-time order
    """
    task_start_times = []

    for machine in schedule_per_machine:
        for batch in schedule_per_machine[machine]:
            for task in batch.tasks:
                task_start_times.append((task, batch.start_time))

    # Sort by start time, then by task id for consistency
    task_start_times.sort(key=lambda x: (x[1], x[0]))
    permutation = np.array([task for task, _ in task_start_times], dtype=np.int_)

    return permutation


def generate_random_permutation(
    problem: "OvenSchedulingProblem",
) -> npt.NDArray[np.int_]:
    """
    Generate a random task permutation.

    Args:
        problem: The problem instance

    Returns:
        Random permutation of task indices
    """
    return np.random.permutation(problem.n_jobs).astype(np.int_)


class VectorOvenSchedulingSolution(Solution):
    """
    Vector-based solution for the Oven Scheduling Problem.

    This solution is encoded as a permutation of tasks, which is decoded
    into a schedule using a greedy decoding scheme. This encoding is suitable
    for metaheuristic algorithms like simulated annealing and genetic algorithms.

    The permutation represents the priority order in which tasks are scheduled.
    A greedy decoder processes tasks in this order and assigns them to machines.

    The decoder can maintain multiple open batches per machine, allowing better
    consolidation of tasks with the same attribute.

    Attributes:
        problem: The OvenSchedulingProblem instance
        permutation: Task permutation (numpy array of task indices)
        _schedule_cache: Cached decoded schedule
        _cache_max_open: Max open batches used for cached schedule
    """

    problem: "OvenSchedulingProblem"

    # Class-level default for decoder configuration
    DEFAULT_MAX_OPEN_BATCHES = 5

    def __init__(
        self,
        problem: "OvenSchedulingProblem",
        permutation: npt.NDArray[np.int_] | None = None,
        schedule_per_machine: dict["Machine", list["ScheduleInfo"]] | None = None,
    ):
        """
        Initialize a vector-based solution.

        Args:
            problem: The problem instance
            permutation: Task permutation. If None, will be extracted from schedule_per_machine
            schedule_per_machine: Decoded schedule. If provided, used to extract permutation
        """
        super().__init__(problem=problem)

        if permutation is None and schedule_per_machine is None:
            raise ValueError(
                "Either permutation or schedule_per_machine must be provided"
            )

        # Store permutation
        if permutation is not None:
            self.permutation = np.array(permutation, dtype=np.int_)
        else:
            self.permutation = extract_permutation_from_schedule(
                problem, schedule_per_machine
            )

        # Lazy evaluation cache
        self._schedule_cache = None
        self._cache_max_open = None

    def __setattr__(self, key, value):
        # Insure that we update the schedule after a job_to_batch change.
        super().__setattr__(key, value)
        if key == "permutation":
            self._schedule_cache = None

    def get_schedule(self, max_open_batches: int | None = None):
        """
        Get the decoded schedule (cached).

        Args:
            max_open_batches: Maximum open batches per machine during decoding.
                            If None, uses DEFAULT_MAX_OPEN_BATCHES.

        Returns:
            OvenSchedulingSolution instance
        """
        from discrete_optimization.ovensched.problem import OvenSchedulingSolution

        if max_open_batches is None:
            max_open_batches = self.DEFAULT_MAX_OPEN_BATCHES

        # Check if cache is valid for this configuration
        if self._schedule_cache is None or self._cache_max_open != max_open_batches:
            schedule_dict = decode_permutation_to_schedule(
                self.problem,
                self.permutation,
                max_open_batches_per_machine=max_open_batches,
            )
            self._schedule_cache = OvenSchedulingSolution(
                problem=self.problem, schedule_per_machine=schedule_dict
            )
            self._cache_max_open = max_open_batches

        return self._schedule_cache

    @property
    def schedule_per_machine(self):
        """
        Get the schedule_per_machine dictionary (delegates to decoded schedule).

        This property makes VectorOvenSchedulingSolution compatible with
        the problem's evaluate() and satisfy() methods.
        """
        return self.get_schedule().schedule_per_machine

    def evaluate(self, max_open_batches: int | None = None) -> dict[str, float]:
        """
        Evaluate the solution.

        Args:
            max_open_batches: Maximum open batches for decoder (uses DEFAULT_MAX_OPEN_BATCHES if None)

        Returns:
            Dictionary of objective values
        """
        schedule = self.get_schedule(max_open_batches=max_open_batches)
        return self.problem.evaluate(schedule)

    def is_feasible(self, max_open_batches: int | None = None) -> bool:
        """
        Check if the solution is feasible.

        Args:
            max_open_batches: Maximum open batches for decoder (uses DEFAULT_MAX_OPEN_BATCHES if None)

        Returns:
            True if feasible, False otherwise
        """
        schedule = self.get_schedule(max_open_batches=max_open_batches)
        return self.problem.satisfy(schedule)

    def copy(self) -> "VectorOvenSchedulingSolution":
        """Create a deep copy of the solution."""
        return VectorOvenSchedulingSolution(
            problem=self.problem,
            permutation=self.permutation.copy(),
        )

    def change_problem(
        self, new_problem: "OvenSchedulingProblem"
    ) -> "VectorOvenSchedulingSolution":
        """
        Change the problem instance.

        Args:
            new_problem: New problem instance

        Returns:
            New solution with the new problem
        """
        return VectorOvenSchedulingSolution(
            problem=new_problem,
            permutation=self.permutation.copy(),
        )

    def to_oven_scheduling_solution(
        self, max_open_batches: int | None = None
    ) -> "OvenSchedulingSolution":
        """
        Convert to standard OvenSchedulingSolution.

        This is useful for compatibility with solvers that expect
        the standard solution format (e.g., CP-SAT warm-start).

        Args:
            max_open_batches: Max open batches for decoder (uses DEFAULT_MAX_OPEN_BATCHES if None)

        Returns:
            OvenSchedulingSolution with decoded schedule
        """
        return self.get_schedule(max_open_batches=max_open_batches)

    def get_summary_string(self, max_open_batches: int | None = None) -> str:
        """
        Generate a human-readable summary of the solution.

        Args:
            max_open_batches: Maximum open batches for decoder (uses DEFAULT_MAX_OPEN_BATCHES if None)

        Returns:
            Formatted string describing the solution
        """
        evaluation = self.evaluate(max_open_batches=max_open_batches)
        feasible = self.is_feasible(max_open_batches=max_open_batches)

        lines = []
        lines.append("=" * 80)
        lines.append("VECTOR OVEN SCHEDULING SOLUTION")
        lines.append("=" * 80)
        lines.append(
            f"\nDecoder config: max_open_batches = {max_open_batches or self.DEFAULT_MAX_OPEN_BATCHES}"
        )
        lines.append(f"Permutation: {self.permutation[:20]}... (showing first 20)")
        lines.append(f"\nObjective Values:")
        lines.append(f"  Processing time: {evaluation['processing_time']}")
        lines.append(f"  Late jobs: {evaluation['nb_late_jobs']}")
        lines.append(f"  Setup cost: {evaluation['setup_cost']}")
        lines.append(f"  Total cost: {sum(evaluation.values())}")
        lines.append(f"\nFeasible: {feasible}")
        lines.append("\n" + "=" * 80)

        return "\n".join(lines)

    def print_summary(self, max_open_batches: int | None = None):
        """Print a human-readable summary of the solution.

        Args:
            max_open_batches: Maximum open batches for decoder (uses DEFAULT_MAX_OPEN_BATCHES if None)
        """
        print(self.get_summary_string(max_open_batches=max_open_batches))

    def __repr__(self) -> str:
        return f"VectorOvenSchedulingSolution(n_jobs={len(self.permutation)})"
