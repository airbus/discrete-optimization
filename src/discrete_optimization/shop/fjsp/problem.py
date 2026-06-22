#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging

from discrete_optimization.shop.base import AnyShopSolution, CommonShopProblem, Task

logger = logging.getLogger(__name__)
NonSkillCumulativeResource = int  # machine id
CumulativeResource = NonSkillCumulativeResource  # no skill
Resource = NonSkillCumulativeResource  # no other resource


FJobShopSolution = AnyShopSolution


class FJobShopProblem(CommonShopProblem):
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
