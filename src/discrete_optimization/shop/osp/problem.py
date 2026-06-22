#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  Open jop problem (no precedence constraint between subjobs, but overlap)
from __future__ import annotations

from discrete_optimization.generic_tasks_tools.calendar_resource import Resource
from discrete_optimization.generic_tasks_tools.multimode_scheduling import (
    SinglemodeSchedulingProblem,
    SinglemodeSchedulingSolution,
)
from discrete_optimization.generic_tasks_tools.precedence import (
    WithoutPrecedenceProblem,
    WithoutPrecedenceSolution,
)
from discrete_optimization.shop.base import AnyShopSolution, CommonShopProblem, Task


class OpenShopSolution(
    AnyShopSolution, SinglemodeSchedulingSolution[Task], WithoutPrecedenceSolution[Task]
):
    problem: OpenShopProblem


class OpenShopProblem(
    CommonShopProblem, SinglemodeSchedulingProblem[Task], WithoutPrecedenceProblem[Task]
):
    def get_task_duration(self, task: Task) -> int:
        return self.list_jobs[task[0]].subjobs[task[1]].recipes[0].processing_time

    def get_resource_availabilities(
        self, resource: Resource
    ) -> list[tuple[int, int, int]]:
        return [(0, self.horizon, 1)]
