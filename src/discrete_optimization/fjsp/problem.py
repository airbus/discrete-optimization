#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from dataclasses import dataclass

from discrete_optimization.generic_tasks_tools.multimode_scheduling import (
    MultimodeSchedulingProblem,
    MultimodeSchedulingSolution,
)
from discrete_optimization.generic_tasks_tools.precedence_scheduling import (
    PrecedenceSchedulingProblem,
    PrecedenceSchedulingSolution,
)
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveDoc,
    ObjectiveHandling,
    ObjectiveRegister,
    Solution,
    TypeObjective,
)
from discrete_optimization.jsp.problem import Subjob, Task

logger = logging.getLogger(__name__)


class FJobShopSolution(
    PrecedenceSchedulingSolution[Task], MultimodeSchedulingSolution[Task]
):
    problem: FJobShopProblem

    def __init__(
        self, problem: FJobShopProblem, schedule: list[list[tuple[int, int, int, int]]]
    ):
        # For each job and sub-job, start, end time, machine id, and option choice given as tuple of int.
        super().__init__(problem=problem)
        self.schedule = schedule

    def copy(self) -> FJobShopSolution:
        return FJobShopSolution(problem=self.problem, schedule=self.schedule)

    def get_end_time(self, task: Task) -> int:
        j, k = task
        return self.schedule[j][k][1]

    def get_start_time(self, task: Task) -> int:
        j, k = task
        return self.schedule[j][k][0]

    def get_machine(self, task: Task) -> int:
        j, k = task
        return self.schedule[j][k][2]

    def get_mode(self, task: Task) -> int:
        """Get 'mode' of given task, aka chosen machine."""
        j, k = task
        return self.schedule[j][k][-1]


SubjobOptions = list[Subjob]


@dataclass
class Job:
    job_id: int
    sub_jobs: list[SubjobOptions]


class FJobShopProblem(
    PrecedenceSchedulingProblem[Task], MultimodeSchedulingProblem[Task]
):
    n_jobs: int
    n_machines: int
    list_jobs: list[Job]

    def __init__(
        self,
        list_jobs: list[Job],
        n_jobs: int = None,
        n_machines: int = None,
        horizon: int = None,
    ):
        self.list_jobs = list_jobs
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.list_jobs = list_jobs
        if self.n_jobs is None:
            self.n_jobs = len(list_jobs)
        if self.n_machines is None:
            self.n_machines = len(
                set(
                    [
                        option.machine_id
                        for job in self.list_jobs
                        for options in job.sub_jobs
                        for option in options
                    ]
                )
            )
        self.n_all_jobs = sum(len(subjob.sub_jobs) for subjob in self.list_jobs)
        self.job_per_machines = {i: [] for i in range(self.n_machines)}
        for k in range(self.n_jobs):
            for sub_k in range(len(list_jobs[k].sub_jobs)):
                for option in range(len(list_jobs[k].sub_jobs[sub_k])):
                    self.job_per_machines[
                        list_jobs[k].sub_jobs[sub_k][option].machine_id
                    ] += [(k, sub_k, option)]
        self.horizon = horizon
        if self.horizon is None:
            self.horizon = sum(
                sum(
                    max(subjob.processing_time for subjob in subjob_opt)
                    for subjob_opt in job.sub_jobs
                )
                for job in self.list_jobs
            )
        self.nb_subjob_per_job = {
            i: len(self.list_jobs[i].sub_jobs) for i in range(self.n_jobs)
        }
        self.subjob_possible_machines = {
            (i, j): set(x.machine_id for x in self.list_jobs[i].sub_jobs[j])
            for i in range(self.n_jobs)
            for j in range(self.nb_subjob_per_job[i])
        }
        self.duration_per_machines = {
            (i, j): {
                x.machine_id: x.processing_time for x in self.list_jobs[i].sub_jobs[j]
            }
            for (i, j) in self.subjob_possible_machines
        }
        self.mode2machine = {
            (j, k): {
                mode: subjob.machine_id for mode, subjob in enumerate(sub_job_options)
            }
            for j, job in enumerate(self.list_jobs)
            for k, sub_job_options in enumerate(job.sub_jobs)
        }
        self.machine2mode = {
            (j, k): {
                subjob.machine_id: mode for mode, subjob in enumerate(sub_job_options)
            }
            for j, job in enumerate(self.list_jobs)
            for k, sub_job_options in enumerate(job.sub_jobs)
        }

    def get_makespan_upper_bound(self) -> int:
        return self.horizon

    def get_task_mode_duration(self, task: Task, mode: int) -> int:
        return self.duration_per_machines[task][self.mode2machine[task][mode]]

    @property
    def tasks_list(self) -> list[Task]:
        return [
            (j, k)
            for j, job in enumerate(self.list_jobs)
            for k in range(len(job.sub_jobs))
        ]

    def get_precedence_constraints(self) -> dict[Task, list[Task]]:
        return {
            (j, k): [(j, k + 1)] if k + 1 < len(job.sub_jobs) else []
            for j, job in enumerate(self.list_jobs)
            for k in range(len(job.sub_jobs))
        }

    def get_task_modes(self, task: Task) -> set[int]:
        j, k = task
        return set(range(len(self.list_jobs[j].sub_jobs[k])))

    def get_last_tasks(self) -> list[Task]:
        return [(j, len(job.sub_jobs) - 1) for j, job in enumerate(self.list_jobs)]

    def evaluate(self, variable: FJobShopSolution) -> dict[str, float]:
        return {"makespan": variable.get_max_end_time()}

    def satisfy(self, variable: FJobShopSolution) -> bool:
        for task, machines in self.subjob_possible_machines.items():
            if not variable.get_machine(task=task) in machines:
                logger.debug(f"Unallowed machine used for task {task}")
                return False
        for m in self.job_per_machines:
            sorted_ = sorted(
                [
                    variable.schedule[x[0]][x[1]]
                    for x in self.job_per_machines[m]
                    if variable.schedule[x[0]][x[1]][2] == m
                ],
                key=lambda y: y[0],
            )
            len_ = len(sorted_)
            for i in range(1, len_):
                if sorted_[i][0] < sorted_[i - 1][1]:
                    logger.debug(f"Overlapping task on machine {m}")
                    return False

        # Check mapping mode <-> machine_id
        for task in self.tasks_list:
            if (
                variable.get_machine(task)
                != self.mode2machine[task][variable.get_mode(task)]
            ):
                logger.debug(
                    f"Machine choice and option choice does not match for task {task}."
                )
                return False

        # Check tasks duration
        if not variable.check_duration_constraints():
            return False

        # Check precedence constraints
        if not variable.check_precedence_constraints():
            return False

        return True

    def get_solution_type(self) -> type[Solution]:
        return FJobShopSolution

    def get_objective_register(self) -> ObjectiveRegister:
        return ObjectiveRegister(
            dict_objective_to_doc={
                "makespan": ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=1)
            },
            objective_sense=ModeOptim.MINIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
        )
