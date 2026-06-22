#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  Job shop model, this was initially implemented in a course material
#  here https://github.com/erachelson/seq_dec_mak/blob/main/scheduling_newcourse/correction/nb2_jobshopsolver.py
from __future__ import annotations

import logging

import wrapt

from discrete_optimization.generic_tasks_tools.allocation import (
    NoUnaryResource,
    WithoutAllocationProblem,
    WithoutAllocationSolution,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling import (
    GenericSchedulingProblem,
    GenericSchedulingSolution,
)
from discrete_optimization.generic_tasks_tools.non_renewable_resource import (
    NoNonRenewableResource,
    WithoutNonRenewableResourceProblem,
    WithoutNonRenewableResourceSolution,
)
from discrete_optimization.generic_tasks_tools.skill import (
    NoSkill,
    WithoutSkillProblem,
    WithoutSkillSolution,
)
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveDoc,
    ObjectiveHandling,
    ObjectiveRegister,
    Solution,
    TypeObjective,
)

logger = logging.getLogger(__name__)


Task = tuple[int, int]
"""Task representation: (job index, subjob index)."""


NonSkillCumulativeResource = int  # machine id
CumulativeResource = NonSkillCumulativeResource  # no skill
Resource = NonSkillCumulativeResource  # no other resource


class SubjobRecipe:
    machine_index: int
    processing_time: int

    def __init__(self, machine_index: int, processing_time: int):
        """Define data of a given subjob"""
        self.machine_index = machine_index
        self.processing_time = processing_time


class Subjob:
    subjob_index: int
    job_index: int
    recipes: list[SubjobRecipe]

    def __init__(self, subjob_index: int, job_index: int, recipes: list[SubjobRecipe]):
        self.subjob_index = subjob_index
        self.job_index = job_index
        self.recipes = recipes


class Job:
    job_index: int
    subjobs: list[Subjob]

    def __init__(self, job_index: int, subjobs: list[Subjob]):
        self.job_index = job_index
        self.subjobs = subjobs


class AnyShopSolution(
    GenericSchedulingSolution[
        Task,
        NoUnaryResource,
        NoSkill,
        NonSkillCumulativeResource,
        NoNonRenewableResource,
    ],
    WithoutSkillSolution[
        Task, NoUnaryResource, NonSkillCumulativeResource, NoUnaryResource
    ],
    WithoutNonRenewableResourceSolution[Task],
    WithoutAllocationSolution[Task],
):
    problem: "CommonShopProblem"
    schedule: list[list[tuple[int, int]]]
    machine_index: list[list[int]]
    recipe_index: list[list[int]]

    def __init__(
        self,
        problem: "CommonShopProblem",
        schedule: list[list[tuple[int, int]]],
        machine_index: list[list[int]] = None,
        recipe_index: list[list[int]] = None,
    ):
        # For each job and sub-job, start, end time, machine id, and option choice given as tuple of int.
        super().__init__(problem=problem)
        self.schedule = schedule
        self.machine_index = machine_index
        if machine_index is None:
            self.machine_index = [
                [
                    self.problem.list_jobs[i].subjobs[j].recipes[0].machine_index
                    for j in range(self.problem.nb_subjob_per_job[i])
                ]
                for i in range(self.problem.n_jobs)
            ]
        self.recipe_index = recipe_index
        if recipe_index is None:
            recipe_index = []
            for i in range(self.problem.n_jobs):
                recipe_job_i = []
                for k in range(len(self.problem.list_jobs[i].subjobs)):
                    machine = self.machine_index[i][k]
                    recipe_job_i.append(
                        self.problem.machine_to_mode_mapping(task=(i, k)).get(
                            machine, None
                        )
                    )
                recipe_index.append(recipe_job_i)
            self.recipe_index = recipe_index

    def copy(self) -> AnyShopSolution:
        return AnyShopSolution(
            problem=self.problem,
            schedule=self.schedule,
            machine_index=self.machine_index,
            recipe_index=self.recipe_index,
        )

    def get_end_time(self, task: Task) -> int:
        j, k = task
        return self.schedule[j][k][1]

    def get_start_time(self, task: Task) -> int:
        j, k = task
        return self.schedule[j][k][0]

    def get_machine(self, task: Task) -> int:
        j, k = task
        return self.machine_index[j][k]

    def get_mode(self, task: Task) -> int:
        """Get 'mode' of given task, aka chosen machine."""
        j, k = task
        return self.recipe_index[j][k]


class CommonShopProblem(
    GenericSchedulingProblem[
        Task,
        NoUnaryResource,
        NoSkill,
        NonSkillCumulativeResource,
        NoNonRenewableResource,
    ],
    WithoutSkillProblem[
        Task, NoUnaryResource, NonSkillCumulativeResource, NoUnaryResource
    ],
    WithoutNonRenewableResourceProblem[Task],
    WithoutAllocationProblem[Task],
):
    n_machines: int
    n_jobs: int
    list_jobs: list[Job]

    def __init__(
        self,
        list_jobs: list[Job],
        n_jobs: int = None,
        n_machines: int = None,
        horizon: int = None,
    ):
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.list_jobs = list_jobs
        if self.n_jobs is None:
            self.n_jobs = len(list_jobs)
        machine_indexes = {
            recipe.machine_index
            for job in self.list_jobs
            for subjob in job.subjobs
            for recipe in subjob.recipes
        }
        if self.n_machines is None:
            self.n_machines = len(machine_indexes)
        assert machine_indexes == set(range(self.n_machines))
        self.n_all_jobs = sum(len(job.subjobs) for job in self.list_jobs)
        # Store
        # for each machine the list of sub-job given as (index_job, index_sub-job, mode-recipe)
        self.job_per_machines = {i: [] for i in range(self.n_machines)}
        for k in range(self.n_jobs):
            for sub_k in range(len(list_jobs[k].subjobs)):
                subjob = list_jobs[k].subjobs[sub_k]
                for index_recipe, recipe in enumerate(subjob.recipes):
                    self.job_per_machines[recipe.machine_index] += [
                        (k, sub_k, index_recipe)
                    ]
        self.horizon = horizon
        if self.horizon is None:
            self.horizon = sum(
                [
                    sum(
                        [
                            max([recipe.processing_time for recipe in subjob.recipes])
                            for subjob in job.subjobs
                        ]
                    )
                    for job in self.list_jobs
                ]
            )

        self.nb_subjob_per_job = {
            i: len(self.list_jobs[i].subjobs) for i in range(self.n_jobs)
        }
        self.subjob_possible_machines = {
            (i, j): set(x.machine_index for x in self.list_jobs[i].subjobs[j].recipes)
            for i in range(self.n_jobs)
            for j in range(self.nb_subjob_per_job[i])
        }
        self.duration_per_machines = {
            (i, j): {
                x.machine_index: x.processing_time
                for x in self.list_jobs[i].subjobs[j].recipes
            }
            for (i, j) in self.subjob_possible_machines
        }
        self.mode2machine = {
            (j, k): self.mode_to_machine_mapping((j, k))
            for j in range(self.n_jobs)
            for k in range(self.nb_subjob_per_job[j])
        }
        self.machine2mode = {
            (j, k): self.machine_to_mode_mapping((j, k))
            for j in range(self.n_jobs)
            for k in range(self.nb_subjob_per_job[j])
        }

    def satisfy(self, variable: AnyShopSolution) -> bool:
        # This check is specific sanity check on the AnyShopSolution
        for task in self.tasks_list:
            if variable.get_mode(task) is None:
                logger.debug(
                    f"Current machine choice is not an allowed option for task {task}"
                )
                return False
            if (
                variable.get_machine(task)
                != self.mode2machine[task][variable.get_mode(task)]
            ):
                logger.debug(
                    f"Machine choice and option choice does not match for task {task}."
                )
                return False
        sat_ = super().satisfy(variable)
        if not sat_:
            logger.debug(f"Automatic check show constraint violation")
            return False
        return True

    def get_makespan_upper_bound(self) -> int:
        return self.horizon

    @wrapt.lru_cache(maxsize=None)
    def machine_to_mode_mapping(self, task: Task):
        return {
            recipe.machine_index: i
            for i, recipe in enumerate(self.list_jobs[task[0]].subjobs[task[1]].recipes)
        }

    @wrapt.lru_cache(maxsize=None)
    def mode_to_machine_mapping(self, task: Task):
        return {
            i: recipe.machine_index
            for i, recipe in enumerate(self.list_jobs[task[0]].subjobs[task[1]].recipes)
        }

    @property
    def non_skill_cumulative_resources_list(self) -> list[NonSkillCumulativeResource]:
        return list(range(self.n_machines))

    def get_cumulative_resource_consumption(
        self, resource: CumulativeResource, task: Task, mode: int
    ) -> int:
        recipe = self.list_jobs[task[0]].subjobs[task[1]].recipes[mode]
        if recipe.machine_index == resource:
            return 1
        else:
            return 0

    @property
    def tasks_list(self) -> list[Task]:
        return [
            (i, j)
            for i in range(self.n_jobs)
            for j in range(len(self.list_jobs[i].subjobs))
        ]

    def get_no_overlap(self) -> set[frozenset[Task]]:
        set_jobs = set()
        for i in range(self.n_jobs):
            set_jobs.add(
                frozenset([(i, j) for j in range(len(self.list_jobs[i].subjobs))])
            )
        return set_jobs

    def get_task_mode_duration(self, task: Task, mode: int) -> int:
        i, j = task
        return self.list_jobs[i].subjobs[j].recipes[mode].processing_time

    def get_task_modes(self, task: Task) -> set[int]:
        return {i for i in range(len(self.list_jobs[task[0]].subjobs[task[1]].recipes))}

    def evaluate(self, variable: AnyShopSolution) -> dict[str, float]:
        return {"makespan": variable.get_max_end_time()}

    def get_solution_type(self) -> type[Solution]:
        return AnyShopSolution

    def get_objective_register(self) -> ObjectiveRegister:
        return ObjectiveRegister(
            dict_objective_to_doc={
                "makespan": ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=1)
            },
            objective_sense=ModeOptim.MINIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
        )
