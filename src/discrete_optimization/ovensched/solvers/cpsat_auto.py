#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import (
    RawSolution,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.auto import (
    GenericSchedulingAutoCpSatSolver,
)
from discrete_optimization.ovensched.problem import (
    OvenSchedulingProblem,
    OvenSchedulingSolution,
    ScheduleInfo,
)

logger = logging.getLogger(__name__)


class OvenSchedulingAutoCpSatSolver(GenericSchedulingAutoCpSatSolver):
    problem: OvenSchedulingProblem

    def retrieve_tasks_variables(self, cpsolvercb):
        """Override to extract batch assignments and batch start/end times."""
        # Call parent to get standard task variables
        for obj in self.dict_objective_expr:
            logger.info(f"{obj} = {cpsolvercb.value(self.dict_objective_expr[obj])}")
        raw_sol = super().retrieve_tasks_variables(cpsolvercb)
        # Extract batch assignments and add to TaskVariable
        for task in self.problem.tasks_list:
            if task in self.task_to_batch_vars:
                for batch_id, var in self.task_to_batch_vars[task].items():
                    if cpsolvercb.Value(var) == 1:
                        raw_sol.task_variables[task].batch_assignment = batch_id
                        break

        # Extract batch entity start/end times
        batch_schedules = {}
        for batch_entity in self.problem.get_batch_entities():
            if batch_entity in self._starts_entity:
                start = cpsolvercb.Value(self._starts_entity[batch_entity])
                end = cpsolvercb.Value(self._ends_entity[batch_entity])
                batch_schedules[batch_entity.batch_id] = (start, end)
        raw_sol.metadata["batch_schedules"] = batch_schedules
        return raw_sol

    def convert_task_variables_to_solution(self, raw_sol: RawSolution):
        """Convert task variables to BatchProcessingSolution."""

        # Extract batch assignments from TaskVariable
        job_to_batch = []
        for task in self.problem.tasks_list:
            task_var = raw_sol.task_variables.get(task)
            if task_var is not None and task_var.batch_assignment is not None:
                job_to_batch.append(task_var.batch_assignment)
            else:
                # Fallback: assign to batch 0 if not found
                job_to_batch.append(0)
        # Extract batch schedules from metadata
        batch_schedules = raw_sol.metadata.get("batch_schedules", {})
        schedule_per_machine = {m: [] for m in range(self.problem.n_machines)}
        # Build schedule_batch: list of (start, end) tuples in sorted batch_id order
        # We need to match the order expected by build_schedule_batch()
        used_batches = sorted(set(job_to_batch))
        schedule_batch = []
        for batch_id in used_batches:
            if batch_id in batch_schedules:
                task_on_this_batch = [
                    self.problem.tasks_list[i]
                    for i in range(len(job_to_batch))
                    if job_to_batch[i] == batch_id
                ]
                if len(task_on_this_batch) > 0:
                    starts = [
                        raw_sol.task_variables[t].start for t in task_on_this_batch
                    ]
                    ends = [raw_sol.task_variables[t].end for t in task_on_this_batch]
                    schedule_batch.append((min(starts), max(ends)))
                    machine = batch_id[0]
                    attr = self.problem.get_task_attribute(task_on_this_batch[0])
                    schedule_per_machine[machine].append(
                        ScheduleInfo(
                            tasks=set(task_on_this_batch),
                            task_attribute=attr,
                            start_time=schedule_batch[-1][0],
                            end_time=schedule_batch[-1][1],
                            machine_batch_index=batch_id,
                        )
                    )

        for m in schedule_per_machine:
            schedule_per_machine[m].sort(key=lambda x: x.start_time)
        # Create solution
        solution = OvenSchedulingSolution(
            problem=self.problem, schedule_per_machine=schedule_per_machine
        )
        solution.batch_schedule = batch_schedules
        return solution
