#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  Job shop model, this was initially implemented in a course material
#  here https://github.com/erachelson/seq_dec_mak/blob/main/scheduling_newcourse/correction/nb2_jobshopsolver.py
from __future__ import annotations

from discrete_optimization.generic_tasks_tools.calendar_resource import Resource
from discrete_optimization.generic_tasks_tools.multimode import (
    SinglemodeSolution,
)
from discrete_optimization.generic_tasks_tools.multimode_scheduling import (
    SinglemodeSchedulingProblem,
)
from discrete_optimization.shop.base import AnyShopSolution, CommonShopProblem, Task


class JobShopSolution(AnyShopSolution, SinglemodeSolution[Task]):
    problem: JobShopProblem


class JobShopProblem(CommonShopProblem, SinglemodeSchedulingProblem[Task]):
    def get_task_duration(self, task: Task) -> int:
        return self.list_jobs[task[0]].subjobs[task[1]].recipes[0].processing_time

    def get_last_tasks(self) -> list[Task]:
        return [
            (j, self.nb_subjob_per_job[j] - 1) for j, job in enumerate(self.list_jobs)
        ]

    def get_resource_availabilities(
        self, resource: Resource
    ) -> list[tuple[int, int, int]]:
        return [(0, self.horizon, 1)]

    def get_precedence_constraints(self) -> dict[Task, list[Task]]:
        return {
            (j, k): [(j, k + 1)] if k + 1 < self.nb_subjob_per_job[j] else []
            for j, job in enumerate(self.list_jobs)
            for k in range(len(job.subjobs))
        }

    def get_no_overlap(self) -> set[frozenset[Task]]:
        # Already taken into account in precedence.
        return {}
        # set_jobs = set()
        # for i in range(self.n_jobs):
        #    set_jobs.add(frozenset([(i, j)
        #                            for j in range(len(self.jobs[i].subjobs))]))
        # return set_jobs
