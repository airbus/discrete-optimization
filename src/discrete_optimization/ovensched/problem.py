#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass

from discrete_optimization.generic_tasks_tools.allocation import (
    AllocationProblem,
    AllocationSolution,
)
from discrete_optimization.generic_tasks_tools.scheduling import (
    SchedulingProblem,
    SchedulingSolution,
)
from discrete_optimization.generic_tools.do_problem import (
    EncodingRegister,
    ModeOptim,
    ObjectiveDoc,
    ObjectiveHandling,
    ObjectiveRegister,
    Solution,
    TypeObjective,
)

logger = logging.getLogger(__name__)
Task = int
TaskAttribute = int
Machine = int
Batch = int
UnaryResource = tuple[Machine, Batch]


@dataclass
class ScheduleInfo:
    """
    Describe the batch
    """

    tasks: set[Task]
    task_attribute: TaskAttribute
    start_time: int
    end_time: int
    machine_batch_index: UnaryResource  # tuple of (machine index, batch index)


class OvenSchedulingSolution(
    SchedulingSolution[Task], AllocationSolution[Task, UnaryResource]
):
    """
    Represents a solution to the Oven Scheduling Problem.
    """

    problem: OvenSchedulingProblem

    def __init__(
        self,
        problem: "OvenSchedulingProblem",
        schedule_per_machine: dict[Machine, list[ScheduleInfo]],
    ):
        """
        Initializes a solution.
        """
        super().__init__(problem)
        self.problem = problem
        self.schedule_per_machine = schedule_per_machine
        self.schedule_per_task: dict[Task, tuple[int, int]] = {}
        for machine in self.schedule_per_machine:
            for batch in self.schedule_per_machine[machine]:
                for t in batch.tasks:
                    self.schedule_per_task[t] = (batch.start_time, batch.end_time)

    def copy(self) -> "OvenSchedulingSolution":
        """Creates a deep copy of the solution."""
        return OvenSchedulingSolution(self.problem, deepcopy(self.schedule_per_machine))

    def get_end_time(self, task: Task) -> int:
        return self.schedule_per_task[task][1]

    def get_start_time(self, task: Task) -> int:
        return self.schedule_per_task[task][0]

    def get_summary_string(self) -> str:
        """Generate a human-readable summary of the solution.

        Returns:
            A formatted string describing the solution
        """
        lines = []
        lines.append("=" * 80)
        lines.append("OVEN SCHEDULING SOLUTION SUMMARY")
        lines.append("=" * 80)

        # Evaluate the solution
        evaluation = self.problem.evaluate(self)
        lines.append("\nObjective Values:")
        lines.append(f"  Processing time: {evaluation['processing_time']}")
        lines.append(f"  Late jobs: {evaluation['nb_late_jobs']}")
        lines.append(f"  Setup cost: {evaluation['setup_cost']}")

        # Feasibility check
        is_feasible = self.problem.satisfy(self)
        lines.append(f"\nSolution is feasible: {is_feasible}")

        # Schedule details
        lines.append(f"\nSchedule Details:")
        total_batches = sum(
            len(batches) for batches in self.schedule_per_machine.values()
        )
        lines.append(f"  Total batches: {total_batches}")
        lines.append(f"  Total tasks: {self.problem.n_jobs}")

        lines.append("\nPer-Machine Schedule:")
        for machine in range(self.problem.n_machines):
            batches = self.schedule_per_machine[machine]
            lines.append(f"\n  Machine {machine}: {len(batches)} batches")

            for i, batch in enumerate(batches):
                lines.append(f"    Batch {i}:")
                lines.append(
                    f"      Tasks: {sorted(batch.tasks)} ({len(batch.tasks)} tasks)"
                )
                lines.append(f"      Attribute: {batch.task_attribute}")
                lines.append(
                    f"      Time window: [{batch.start_time}, {batch.end_time}]"
                )
                lines.append(f"      Duration: {batch.end_time - batch.start_time}")

                # Calculate setup for this batch
                if i == 0:
                    prev_attr = self.problem.machines_data[machine].initial_attribute
                else:
                    prev_attr = batches[i - 1].task_attribute
                setup_time = self.problem.setup_times[prev_attr][batch.task_attribute]
                setup_cost = self.problem.setup_costs[prev_attr][batch.task_attribute]
                lines.append(
                    f"      Setup time: {setup_time}, Setup cost: {setup_cost}"
                )

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)

    def print_summary(self):
        """Print a human-readable summary of the solution."""
        print(self.get_summary_string())

    def is_allocated(self, task: Task, unary_resource: UnaryResource) -> bool:
        machine, index = unary_resource
        if index < len(self.schedule_per_machine[machine]):
            return task in self.schedule_per_machine[machine][index].tasks
        return False


@dataclass
class TaskData:
    attribute: TaskAttribute
    min_duration: int
    max_duration: int
    earliest_start: int
    latest_end: int
    eligible_machines: set[Machine]
    size: int


@dataclass
class MachineData:
    capacity: int
    initial_attribute: TaskAttribute
    availability: list[tuple[int, int]]


class OvenSchedulingProblem(
    SchedulingProblem[Task], AllocationProblem[Task, UnaryResource]
):
    """Defines an instance of the Oven Scheduling Problem (OSP) and its evaluation logic."""

    def get_makespan_upper_bound(self) -> int:
        return 100000

    @property
    def unary_resources_list(self) -> list[UnaryResource]:
        return [
            (machine, task)
            for machine in range(self.n_machines)
            for task in range(self.n_jobs)
        ]

    @property
    def tasks_list(self) -> list[Task]:
        return list(range(self.n_jobs))

    def get_set_task_attributes(self) -> set[TaskAttribute]:
        return set([td.attribute for td in self.tasks_data])

    def __init__(
        self,
        n_jobs: int,
        n_machines: int,
        tasks_data: list[TaskData],
        machines_data: list[MachineData],
        setup_costs: list[list[int]],
        setup_times: list[list[int]],
    ):
        """Initializes the problem with all its parameters."""
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.tasks_data = tasks_data
        self.machines_data = machines_data
        self.fix_availability_intervals()
        self.setup_costs = setup_costs
        self.setup_times = setup_times
        self.additional_data = {}

    def fix_availability_intervals(self):
        for m_data in self.machines_data:
            fixed_m = [m_data.availability[0]]
            for slot in m_data.availability[1:]:
                if slot[0] == fixed_m[-1][1]:
                    fixed_m[-1] = (fixed_m[-1][0], slot[1])
                else:
                    fixed_m.append(slot)
            m_data.availability = fixed_m

    def get_solution_type(self) -> type[Solution]:
        """Returns the solution class type."""
        return OvenSchedulingSolution

    def get_objective_register(self) -> ObjectiveRegister:
        """Defines the problem's objectives."""
        if "additional_data" in self.__dict__.keys() and self.additional_data:
            add = self.additional_data
            return ObjectiveRegister(
                objective_sense=ModeOptim.MINIMIZATION,
                objective_handling=ObjectiveHandling.AGGREGATE,
                dict_objective_to_doc={
                    "processing_time": ObjectiveDoc(
                        type=TypeObjective.OBJECTIVE,
                        default_weight=add.get("weight_processing", 1),
                    ),
                    "nb_late_jobs": ObjectiveDoc(
                        type=TypeObjective.OBJECTIVE,
                        default_weight=add.get("weight_tardiness", 1),
                    ),
                    "setup_cost": ObjectiveDoc(
                        type=TypeObjective.OBJECTIVE,
                        default_weight=add.get("weight_setup_cost", 1),
                    ),
                },
            )
        else:
            return ObjectiveRegister(
                objective_sense=ModeOptim.MINIMIZATION,
                objective_handling=ObjectiveHandling.AGGREGATE,
                dict_objective_to_doc={
                    "processing_time": ObjectiveDoc(
                        type=TypeObjective.OBJECTIVE, default_weight=1
                    ),
                    "nb_late_jobs": ObjectiveDoc(
                        type=TypeObjective.OBJECTIVE, default_weight=1
                    ),
                    "setup_cost": ObjectiveDoc(
                        type=TypeObjective.OBJECTIVE, default_weight=1
                    ),
                },
            )

    def get_attribute_register(self) -> EncodingRegister:
        """Defines the solution's encoding."""
        return EncodingRegister(dict_attribute_to_type={})

    def satisfy(self, variable: OvenSchedulingSolution) -> bool:
        """
        Checks if a solution is feasible by verifying static and temporal constraints.

        If a timed_schedule is already cached in the solution, this function will
        validate the cached timings. Otherwise, it will attempt to compute them.
        """
        # 1. Perform all static checks first (job uniqueness, attributes, capacity, etc.)
        all_jobs = set(range(self.n_jobs))
        scheduled_jobs = set()
        for m in variable.schedule_per_machine:
            capacity = self.machines_data[m].capacity
            for i_batch in range(len(variable.schedule_per_machine[m])):
                schedule_info: ScheduleInfo = variable.schedule_per_machine[m][i_batch]
                batch_tasks = schedule_info.tasks
                # 1. Check if tasks are not already scheduled
                if not batch_tasks.isdisjoint(scheduled_jobs):
                    logger.info("Some tasks are scheduled several time")
                    return False
                scheduled_jobs.update(batch_tasks)
                attribute_batch = schedule_info.task_attribute
                attributes = [self.tasks_data[j].attribute for j in batch_tasks]
                # 2. Check that all the tasks in the batch have the same attribute
                if not all(a == attribute_batch for a in attributes):
                    logger.info(f"Different attributes in a given batch {m, i_batch}")
                    return False
                # 3. Check capacity of the batch.
                if sum(self.tasks_data[j].size for j in batch_tasks) > capacity:
                    logger.info(f"Capacity of the {m, i_batch} batch is broken")
                    return False
                # 4. Check task eligibility to the machine
                for task in batch_tasks:
                    if m not in self.tasks_data[task].eligible_machines:
                        logger.info(f"Machine {m} is not eligible for task {task}")
                        return False
                # 5. Check duration compatibility inside the batch
                if max(self.tasks_data[j].min_duration for j in batch_tasks) > min(
                    self.tasks_data[j].max_duration for j in batch_tasks
                ):
                    logger.info(f"Duration incompatible in the {m, i_batch} batch")
                    return False
                # 6. Check if batch duration is coherent with the tasks inside :
                dur = schedule_info.end_time - schedule_info.start_time
                if dur > min(self.tasks_data[j].max_duration for j in batch_tasks):
                    logger.info(f"Duration of batch {m, i_batch} is too high")
                    return False
                if dur < max(self.tasks_data[j].min_duration for j in batch_tasks):
                    logger.info(
                        f"Duration of batch {m, i_batch} is too low for one task in the batch"
                    )
                    return False
                # 7. Check if the batch is done at an availability interval.
                start_time = schedule_info.start_time
                end_time = schedule_info.end_time
                is_in_valid_window = any(
                    avail_start <= start_time and end_time <= avail_end
                    for avail_start, avail_end in self.machines_data[m].availability
                )
                if not is_in_valid_window:
                    print(start_time, end_time, self.machines_data[m].availability)
                    logger.info(
                        f"Batch {m, i_batch} is not in valid window for machine {m}"
                    )
                    return False
                if any(
                    start_time < self.tasks_data[j].earliest_start for j in batch_tasks
                ):
                    logger.info(
                        f"Batch {m, i_batch} starts earlier than one task earliest start"
                    )
                    return False
                if i_batch >= 1:
                    prev_batch_attribute = variable.schedule_per_machine[m][
                        i_batch - 1
                    ].task_attribute
                    prev_time = variable.schedule_per_machine[m][i_batch - 1].end_time
                else:
                    prev_batch_attribute = self.machines_data[m].initial_attribute
                    prev_time = 0
                setup_time = self.setup_times[prev_batch_attribute][attribute_batch]
                if start_time < prev_time + setup_time:
                    logger.info(
                        f"Batch {m, i_batch} starts earlier previous batch + setup time"
                    )
                    return False
        if scheduled_jobs != all_jobs:
            logger.info(f"Some jobs are missing")
            return False
        return True

    def evaluate(self, variable: OvenSchedulingSolution) -> dict[str, float]:
        """Evaluates a solution's performance."""
        nb_late_tasks = 0
        processing_time = 0
        setup_cost = 0
        for m in variable.schedule_per_machine:
            for i_batch in range(len(variable.schedule_per_machine[m])):
                batch = variable.schedule_per_machine[m][i_batch]
                start = batch.start_time
                end = batch.end_time
                for task in batch.tasks:
                    if end > self.tasks_data[task].latest_end:
                        nb_late_tasks += 1
                processing_time += end - start
                attribute_i_batch = batch.task_attribute
                if i_batch >= 1:
                    prev_attribute = variable.schedule_per_machine[m][
                        i_batch - 1
                    ].task_attribute
                else:
                    prev_attribute = self.machines_data[m].initial_attribute
                setup_cost += self.setup_costs[prev_attribute][attribute_i_batch]
        return {
            "processing_time": processing_time,
            "nb_late_jobs": nb_late_tasks,
            "setup_cost": setup_cost,
        }
