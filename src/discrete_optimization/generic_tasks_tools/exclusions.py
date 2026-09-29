#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  Problem implementation of resource exclusion constraint
#  This means for that some task are blocking some area, that other task might need.
#  This differs from classical cumulative resource constraint.
#     - Several blocking tasks might overlap on the same resource
#     (so we cant just model full resource consumption of those tasks)
#     - The capacity might be different from 1, so if the resource is not blocked by blocking task
#     we might have multiple task on the resource.
import logging
from typing import Generic, Hashable, TypeVar

import numpy as np

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.multimode import MultimodeSolution
from discrete_optimization.generic_tasks_tools.multimode_scheduling import (
    MultimodeSchedulingProblem,
)
from discrete_optimization.generic_tasks_tools.scheduling import SchedulingSolution
from discrete_optimization.generic_tasks_tools.utils import optional_override

logger = logging.getLogger(__name__)
ExclusionResource = TypeVar("ExclusionResource", bound=Hashable)


class ExclusionProblem(
    MultimodeSchedulingProblem[Task], Generic[Task, ExclusionResource]
):
    @property
    @optional_override
    def exclusion_resources_list(self) -> list[ExclusionResource]:
        return []

    def has_exclusion_resource(self):
        return len(self.exclusion_resources_list) > 0

    @optional_override
    def is_task_mode_excluding_others(
        self, task: Task, mode: int, resource: ExclusionResource
    ) -> bool:
        return False

    @optional_override
    def is_task_always_excluding_others(
        self, task: Task, resource: ExclusionResource
    ) -> bool:
        return all(
            self.is_task_mode_excluding_others(task, mode, resource)
            for mode in self.get_task_modes(task)
        )

    @optional_override
    def is_task_possibly_excluding_others(
        self, task: Task, resource: ExclusionResource
    ) -> bool:
        return any(
            self.is_task_mode_excluding_others(task, mode, resource)
            for mode in self.get_task_modes(task)
        )

    @optional_override
    def get_tasks_modes_exclude_others(
        self, resource: ExclusionResource
    ) -> set[tuple[Task, int]]:
        return {
            (task, mode)
            for task in self.tasks_list
            for mode in self.get_task_modes(task)
            if self.is_task_mode_excluding_others(task, mode, resource)
        }

    @optional_override
    def get_tasks_always_exclude_others(self, resource: ExclusionResource) -> set[Task]:
        return {
            task
            for task in self.tasks_list
            if self.is_task_always_excluding_others(task, resource)
        }

    @optional_override
    def get_tasks_possibly_exclude_others(
        self, resource: ExclusionResource
    ) -> set[Task]:
        return {
            task
            for task in self.tasks_list
            if self.is_task_possibly_excluding_others(task, resource)
        }

    @optional_override
    def get_task_consumption_exclusion_resource(
        self, resource: ExclusionResource, task: Task, mode: int
    ):
        return 0

    def get_tasks_modes_consuming_exclusion_resource(
        self, resource: ExclusionResource
    ) -> set[tuple[Task, int, int]]:
        return {
            (task, mode, conso)
            for task in self.tasks_list
            for mode in self.get_task_modes(task)
            if (
                conso := self.get_task_consumption_exclusion_resource(
                    resource=resource, task=task, mode=mode
                )
            )
            > 0
        }

    @optional_override
    def get_tasks_possibly_excluded(self, resource: ExclusionResource) -> set[Task]:
        set_task_mode_conso = self.get_tasks_modes_consuming_exclusion_resource(
            resource
        )
        return {t[0] for t in set_task_mode_conso}

    @optional_override
    def get_capacity_exclusion_resource(self, resource: ExclusionResource):
        return 1


class ExclusionSolution(
    SchedulingSolution[Task], MultimodeSolution[Task], Generic[Task, ExclusionResource]
):
    problem: ExclusionProblem[Task, ExclusionResource]

    def check_exclusion_constraint(self):
        if not self.problem.has_exclusion_resource():
            return True
        max_time = self.get_max_end_time()
        for r in self.problem.exclusion_resources_list:
            capacity = self.problem.get_capacity_exclusion_resource(r)
            blocked = np.zeros(max_time, dtype=np.bool)
            # Check the cumulative constraint too.
            consumption = np.zeros(max_time, dtype=int)
            task_mode_excluding = self.problem.get_tasks_modes_exclude_others(r)
            task_mode_consuming = (
                self.problem.get_tasks_modes_consuming_exclusion_resource(r)
            )
            for t, m in task_mode_excluding:
                if self.is_present(t) and self.get_mode(t) == m:
                    blocked[self.get_start_time(t) : self.get_end_time(t)] = True
            for t, m, c in task_mode_consuming:
                if self.is_present(t) and self.get_mode(t) == m:
                    st, end = self.get_start_time(t), self.get_end_time(t)
                    if end == st:
                        continue
                    if np.max(blocked[st:end]) == 1:
                        logger.info(
                            f"Task {t} in mode {m} is running during an excluded time "
                            f"of resource {r}"
                            f"{blocked[st:end]}"
                        )
                        return False
                    consumption[st:end] += c
                    if np.max(consumption[st:end]) > capacity:
                        logger.info(
                            f"Capacity of exclusion resource {r} "
                            f"is over-reached between {st} and {end}: {capacity} vs {consumption[st:end]}"
                        )
                        return False
        return True
