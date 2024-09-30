#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import List

from discrete_optimization.generic_tools.do_problem import *
from discrete_optimization.jsp.job_shop_problem import Subjob


class SolutionFJobshop(Solution):
    def __init__(
        self, problem: "FJobShopProblem", schedule: list[list[tuple[int, int, int]]]
    ):
        # For each job and sub-job, start, end time and machine id choice given as tuple of int.
        self.problem = problem
        self.schedule = schedule

    def copy(self) -> "Solution":
        return SolutionFJobshop(problem=self.problem, schedule=self.schedule)

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

    def evaluate(self, variable: SolutionFJobshop) -> dict[str, float]:
        return {"makespan": max(x[-1][1] for x in variable.schedule)}

    def satisfy(self, variable: Solution) -> bool:
        return True

    def get_attribute_register(self) -> EncodingRegister:
        return EncodingRegister(dict_attribute_to_type={})

    def get_solution_type(self) -> type[Solution]:
        return SolutionFJobshop

    def get_objective_register(self) -> ObjectiveRegister:
        return ObjectiveRegister(
            dict_objective_to_doc={
                "makespan": ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=1)
            },
            objective_sense=ModeOptim.MINIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
        )
