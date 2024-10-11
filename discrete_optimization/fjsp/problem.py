#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import List

from discrete_optimization.generic_tools.do_problem import *
from discrete_optimization.jsp.problem import Subjob


class FJobShopSolution(Solution):
    def __init__(
        self, problem: "FJobShopProblem", schedule: list[list[tuple[int, int, int]]]
    ):
        # For each job and sub-job, start, end time and machine id choice given as tuple of int.
        self.problem = problem
        self.schedule = schedule

    def copy(self) -> "Solution":
        return FJobShopSolution(problem=self.problem, schedule=self.schedule)

    def change_problem(self, new_problem: "Problem") -> None:
        self.problem = new_problem


SubjobOptions = list[Subjob]


@dataclass
class Job:
    job_id: int
    sub_jobs: List[SubjobOptions]


class FJobShopProblem(Problem):
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

    def evaluate(self, variable: FJobShopSolution) -> dict[str, float]:
        return {"makespan": max(x[-1][1] for x in variable.schedule)}

    def satisfy(self, variable: FJobShopSolution) -> bool:
        if not all(
            variable.schedule[i][j][2] in self.subjob_possible_machines[(i, j)]
            for (i, j) in self.subjob_possible_machines
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
            m = variable.schedule[job][0][2]
            if not (
                variable.schedule[job][0][1] - variable.schedule[job][0][0]
                == self.duration_per_machines[(job, 0)][m]
            ):
                logger.info(
                    f"Duration of task {job, 0} not coherent with the machine choice "
                )
            for s_j in range(1, len(variable.schedule[job])):
                if variable.schedule[job][s_j][0] < variable.schedule[job][s_j - 1][1]:
                    logger.info(
                        f"Precedence constraint not respected between {job, s_j}"
                        f"and {job, s_j-1}"
                    )
                    return False
                if not (
                    variable.schedule[job][s_j][1] - variable.schedule[job][s_j][0]
                    == self.duration_per_machines[(job, s_j)][
                        variable.schedule[job][s_j][2]
                    ]
                ):
                    logger.info(
                        f"Duration of task {job, s_j} not coherent with the machine choice "
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
