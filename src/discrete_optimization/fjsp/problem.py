#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from dataclasses import dataclass

from discrete_optimization.generic_tasks_tools.multimode import (
    MultimodeProblem,
    MultimodeSolution,
)
from discrete_optimization.generic_tasks_tools.precedence import PrecedenceProblem
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
from discrete_optimization.jsp.problem import Subjob, Task

logger = logging.getLogger(__name__)


class FJobShopSolution(SchedulingSolution[Task], MultimodeSolution[Task]):
    problem: FJobShopProblem

    def __init__(
        self, problem: FJobShopProblem, schedule: list[list[tuple[int, int, int, int]]]
    ):
        # For each job and sub-job, start, end time, machine id, and option choice given as tuple of int.
        self.problem = problem
        self.schedule = schedule

    def copy(self) -> FJobShopSolution:
        return FJobShopSolution(problem=self.problem, schedule=self.schedule)

    def change_problem(self, new_problem: FJobShopProblem) -> None:
        self.problem = new_problem

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
    SchedulingProblem[Task], MultimodeProblem[Task], PrecedenceProblem[Task]
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

    def get_makespan_upper_bound(self) -> int:
        return self.horizon

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
        if not all(
            variable.get_machine(task=task) in machines
            for task, machines in self.subjob_possible_machines.items()
        ):
            logger.info("Unallowed machine used for some subjob")
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
                    logger.info("Overlapping task on same machines")
                    return False
        for job in range(self.n_jobs):
            s_j = 0
            i_opt = variable.schedule[job][s_j][-1]
            machine_id = variable.schedule[job][s_j][2]
            if self.list_jobs[job].sub_jobs[s_j][i_opt].machine_id != machine_id:
                logger.info(
                    f"Machine choice and option choice does not match for task {job, s_j}."
                )
                return False
            if not (
                variable.schedule[job][s_j][1] - variable.schedule[job][s_j][0]
                == self.duration_per_machines[(job, s_j)][machine_id]
            ):
                logger.info(
                    f"Duration of task {job, s_j} not coherent with the machine choice "
                )
            for s_j in range(1, len(variable.schedule[job])):
                if variable.schedule[job][s_j][0] < variable.schedule[job][s_j - 1][1]:
                    logger.info(
                        f"Precedence constraint not respected between {job, s_j}"
                        f"and {job, s_j - 1}"
                    )
                    return False
                machine_id = variable.schedule[job][s_j][2]
                if not (
                    variable.schedule[job][s_j][1] - variable.schedule[job][s_j][0]
                    == self.duration_per_machines[(job, s_j)][machine_id]
                ):
                    logger.info(
                        f"Duration of task {job, s_j} not coherent with the machine choice "
                    )
                    return False
                i_opt = variable.schedule[job][s_j][-1]
                if self.list_jobs[job].sub_jobs[s_j][i_opt].machine_id != machine_id:
                    logger.info(
                        f"Machine choice and option choice does not match for task {job, s_j}."
                    )
                    return False

        return True

    def get_attribute_register(self) -> EncodingRegister:
        return EncodingRegister(dict_attribute_to_type={})

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
