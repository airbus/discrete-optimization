#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Greedy heuristic solver for Oven Scheduling Problem."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.do_solver import SolverDO, StatusSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.ovensched.problem import (
    OvenSchedulingProblem,
    OvenSchedulingSolution,
    ScheduleInfo,
)

logger = logging.getLogger(__name__)


class GreedyOvenSchedulingSolver(SolverDO):
    """Greedy heuristic solver for Oven Scheduling.

    Strategy: Group jobs by attribute, greedily pack into batches respecting
    capacity, duration compatibility, and machine availability windows.
    """

    problem: OvenSchedulingProblem

    def __init__(
        self,
        problem: OvenSchedulingProblem,
        params_objective_function: ParamsObjectiveFunction | None = None,
        **kwargs: Any,
    ):
        """Initialize the greedy solver.

        Args:
            problem: The oven scheduling problem instance
            params_objective_function: Parameters for objective function
            **kwargs: Additional arguments
        """
        super().__init__(problem, params_objective_function, **kwargs)

    def init_model(self, **kwargs: Any) -> None:
        """Initialize the model (no-op for greedy heuristic)."""
        pass

    def solve(self, **kwargs: Any) -> ResultStorage:
        """Solve the problem using greedy heuristic.

        Returns:
            ResultStorage with the greedy solution
        """
        logger.info("Running greedy heuristic solver...")
        solution = self._greedy_by_attribute_batching()
        # Evaluate solution
        fitness = self.aggreg_from_sol(solution)
        # Create result storage
        result = self.create_result_storage(
            list_solution_fits=[(solution, fitness)],
        )
        logger.info(f"Greedy solution: fitness={fitness}")
        self.status_solver = StatusSolver.SATISFIED
        return result

    def _greedy_by_attribute_batching(self) -> OvenSchedulingSolution:
        """Greedy heuristic: group jobs by attribute, create batches greedily.

        Strategy:
        1. For each machine, process attributes in order
        2. For each attribute, greedily pack jobs into batches respecting capacity
        3. Choose batch duration as middle of compatible duration ranges
        4. Respect machine availability windows

        Returns:
            OvenSchedulingSolution
        """
        schedule_per_machine = {m: [] for m in range(self.problem.n_machines)}
        scheduled = set()

        # Group jobs by (attribute, eligible machines)
        jobs_by_attr_machine = defaultdict(list)
        for job_id in range(self.problem.n_jobs):
            task = self.problem.tasks_data[job_id]
            attr = task.attribute
            for machine in task.eligible_machines:
                jobs_by_attr_machine[(attr, machine)].append(job_id)

        # For each machine, process attributes and create batches
        for m in range(self.problem.n_machines):
            capacity = self.problem.machines_data[m].capacity
            availability_windows = self.problem.machines_data[m].availability

            # Get all attributes that have jobs eligible for this machine
            attrs_on_machine = sorted(
                set(a for a, mach in jobs_by_attr_machine.keys() if mach == m)
            )

            for attr in attrs_on_machine:
                available_jobs = [
                    j
                    for j in jobs_by_attr_machine.get((attr, m), [])
                    if j not in scheduled
                ]

                while available_jobs:
                    # Start new batch
                    batch_jobs = []
                    batch_size = 0
                    batch_dur_min = 0
                    batch_dur_max = float("inf")

                    # Greedily pack jobs into batch
                    for job_id in sorted(available_jobs):
                        task = self.problem.tasks_data[job_id]

                        if batch_size + task.size <= capacity:
                            # Check duration compatibility
                            new_min = max(batch_dur_min, task.min_duration)
                            new_max = min(batch_dur_max, task.max_duration)

                            if new_min <= new_max:
                                batch_jobs.append(job_id)
                                batch_size += task.size
                                batch_dur_min = new_min
                                batch_dur_max = new_max

                    if not batch_jobs:
                        break

                    # Choose duration (middle of valid range)
                    batch_duration = (batch_dur_min + batch_dur_max) // 2

                    # Compute start time (considering previous batches and setup cost)
                    candidate_start = 0
                    if schedule_per_machine[m]:
                        last_batch = schedule_per_machine[m][-1]
                        candidate_start = last_batch.end_time
                        # Add setup cost
                        prev_attr = last_batch.task_attribute
                        setup_cost = self.problem.setup_costs[prev_attr][attr]
                        candidate_start += setup_cost

                    # Find next available window that can fit this batch
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

                    # If no window found, skip these jobs
                    if start_time is None:
                        break

                    # Create batch
                    batch_idx = len(schedule_per_machine[m])
                    schedule_per_machine[m].append(
                        ScheduleInfo(
                            tasks=set(batch_jobs),
                            task_attribute=attr,
                            start_time=start_time,
                            end_time=end_time,
                            machine_batch_index=(m, batch_idx),
                        )
                    )

                    # Mark as scheduled
                    for job_id in batch_jobs:
                        scheduled.add(job_id)
                        available_jobs.remove(job_id)

        return OvenSchedulingSolution(
            problem=self.problem,
            schedule_per_machine=schedule_per_machine,
        )
