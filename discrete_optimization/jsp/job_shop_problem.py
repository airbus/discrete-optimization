#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  Job shop model, this was initially implemented in a course material
#  here https://github.com/erachelson/seq_dec_mak/blob/main/scheduling_newcourse/correction/nb2_jobshopsolver.py


from discrete_optimization.generic_tools.do_problem import *


class SolutionJobshop(Solution):
    def __init__(
        self, problem: "JobShopProblem", schedule: list[list[tuple[int, int]]]
    ):
        # For each job and sub-job, start and end time given as tuple of int.
        self.problem = problem
        self.schedule = schedule

    def copy(self) -> "Solution":
        return SolutionJobshop(problem=self.problem, schedule=self.schedule)

    def change_problem(self, new_problem: "Problem") -> None:
        self.problem = new_problem


class Subjob:
    machine_id: int
    processing_time: int

    def __init__(self, machine_id: int, processing_time: int):
        """Define data of a given subjob"""
        self.machine_id = machine_id
        self.processing_time = processing_time


class JobShopProblem(Problem):
    n_jobs: int
    n_machines: int
    list_jobs: list[list[Subjob]]

    def __init__(
        self, list_jobs: list[list[Subjob]], n_jobs: int = None, n_machines: int = None
    ):
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.list_jobs = list_jobs
        if self.n_jobs is None:
            self.n_jobs = len(list_jobs)
        if self.n_machines is None:
            self.n_machines = len(
                set([y.machine_id for x in self.list_jobs for y in x])
            )
        self.n_all_jobs = sum(len(subjob) for subjob in self.list_jobs)
        # Store for each machine the list of sub-job given as (index_job, index_sub-job)
        self.job_per_machines = {i: [] for i in range(self.n_machines)}
        for k in range(self.n_jobs):
            for sub_k in range(len(list_jobs[k])):
                self.job_per_machines[list_jobs[k][sub_k].machine_id] += [(k, sub_k)]

    def evaluate(self, variable: SolutionJobshop) -> dict[str, float]:
        return {"makespan": max(x[-1][1] for x in variable.schedule)}

    def satisfy(self, variable: SolutionJobshop) -> bool:
        for m in self.job_per_machines:
            sorted_ = sorted(
                [variable.schedule[x[0]][x[1]] for x in self.job_per_machines[m]],
                key=lambda y: y[0],
            )
            for i in range(1, len(sorted_)):
                if sorted_[i][0] < sorted_[i - 1][1]:
                    return False
        for job in range(self.n_jobs):
            for s_j in range(1, len(variable.schedule[job])):
                if variable.schedule[job][s_j][0] < variable.schedule[job][s_j - 1][1]:
                    return False
        return True

    def get_attribute_register(self) -> EncodingRegister:
        return EncodingRegister(dict_attribute_to_type={})

    def get_solution_type(self) -> type[Solution]:
        return SolutionJobshop

    def get_objective_register(self) -> ObjectiveRegister:
        return ObjectiveRegister(
            dict_objective_to_doc={
                "makespan": ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=1)
            },
            objective_sense=ModeOptim.MINIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
        )
