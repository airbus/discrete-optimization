#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Generic

from discrete_optimization.generic_tasks_tools.allocation import (
    MultimodeAllocationProblem,
    MultiModeAllocationSolution,
    Task,
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.calendar_resource import (
    Resource,
)
from discrete_optimization.generic_tasks_tools.cumulative_resource import (
    CumulativeResource,
    CumulativeResourceProblem,
    CumulativeResourceSolution,
)
from discrete_optimization.generic_tasks_tools.utils import optional_override


class CumulativeResourceUsageByUnaryResourceProblem(
    MultimodeAllocationProblem[Task, UnaryResource],
    CumulativeResourceProblem[Task, CumulativeResource, UnaryResource],
    Generic[Task, CumulativeResource, UnaryResource],
):
    @optional_override
    def get_resource_consumption_when_unary_resource_allocated(
        self, task: Task, mode: int, resource: Resource, unary_resource: UnaryResource
    ):
        return 0

    def has_any_resource_consumption_depend_on_unary_resource(self):
        return any(
            len(self.get_non_zero_mode_res_unary(task)) > 0 for task in self.tasks_list
        )

    def get_tasks_of_interest_for_resource(self, resource: Resource):
        return set(
            [
                t
                for t in self.tasks_list
                if any(
                    self.get_resource_consumption_when_unary_resource_allocated(
                        task=t, mode=m, resource=resource, unary_resource=ur
                    )
                    > 0
                    for m in self.get_task_modes(t)
                    for ur in self.unary_resources_list
                )
            ]
        )

    def get_non_zero_mode_res_unary(
        self, task: Task
    ) -> list[tuple[int, Resource, UnaryResource]]:
        return [
            (mode, res, unary)
            for mode in self.get_task_modes(task)
            for res in self.cumulative_resources_list
            for unary in self.unary_resources_list
            if self.get_resource_consumption_when_unary_resource_allocated(
                task, mode, res, unary
            )
            > 0
        ]


class CumulativeResourceUsageByUnarySolution(
    MultiModeAllocationSolution[Task, UnaryResource],
    CumulativeResourceSolution[Task, CumulativeResource, UnaryResource],
):
    problem: CumulativeResourceUsageByUnaryResourceProblem[
        Task, CumulativeResource, UnaryResource
    ]

    def get_resource_consumption_by_unary_allocation(
        self, task: Task, resource: CumulativeResource
    ):
        if not self.is_present(task):
            return 0
        mode = self.get_mode(task)
        allocated = self.get_task_allocation(task)
        value = 0
        for unary in allocated:
            value += (
                self.problem.get_resource_consumption_when_unary_resource_allocated(
                    task=task, mode=mode, resource=resource, unary_resource=unary
                )
            )
        return value
