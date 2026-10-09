#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Generic, Iterable

import numpy as np

from discrete_optimization.generic_tasks_tools.calendar_preemptive import (
    CalendarPreemptiveProblem,
    CalendarPreemptiveSolution,
    CumulativeResource,
    Task,
)
from discrete_optimization.generic_tasks_tools.cumulative_resource import (
    CumulativeResource,
    Resource,
)
from discrete_optimization.generic_tasks_tools.resource_usage_by_unary import (
    CumulativeResourceUsageByUnaryResourceProblem,
    CumulativeResourceUsageByUnarySolution,
    UnaryResource,
)


class CumulativeResourceGenericProblem(
    CalendarPreemptiveProblem[Task, CumulativeResource, UnaryResource],
    CumulativeResourceUsageByUnaryResourceProblem[
        Task, CumulativeResource, UnaryResource
    ],
    Generic[Task, CumulativeResource, UnaryResource],
):
    pass


class CumulativeResourceGenericSolution(
    CalendarPreemptiveSolution[Task, CumulativeResource, UnaryResource],
    CumulativeResourceUsageByUnarySolution[Task, CumulativeResource, UnaryResource],
    Generic[Task, CumulativeResource, UnaryResource],
):
    problem: CumulativeResourceGenericProblem[Task, CumulativeResource, UnaryResource]

    def _compute_calendar_resource_consumption_np(
        self, resources: Iterable[Resource]
    ) -> np.ndarray:
        if (
            not self.problem.has_any_calendar_preempted()
            and not self.problem.has_any_resource_consumption_depend_on_unary_resource()
        ):
            # Fallback to CalendarResource checker.
            return super()._compute_calendar_resource_consumption_np(resources)
        # Override the util function, so that the cumulative calendar resource constraint
        # is well checked ! we remove the consumption of the task on its idle time.
        makespan = self.get_max_end_time()
        resources_consumption = {
            resource: np.zeros(makespan, dtype=int) for resource in resources
        }
        for task in self.get_present_tasks():
            start = self.get_start_time(task)
            end = self.get_end_time(task)
            mode = self.get_mode(task)
            if task not in self.problem.get_all_tasks_calendar_preempted():
                for resource in resources:
                    val = self.get_calendar_resource_consumption(
                        resource=resource, task=task
                    )
                    val += self.get_resource_consumption_by_unary_allocation(
                        task=task, resource=resource
                    )
                    resources_consumption[resource][start:end] += val
            else:
                mask = self.problem.calendar_preemption_data.get_binary_calendar(
                    task, mode
                )
                for resource in resources:
                    resources_consumption[resource][start:end] += np.multiply(
                        mask[start:end],
                        self.get_calendar_resource_consumption(
                            resource=resource, task=task
                        ),
                    )
        return resources_consumption
