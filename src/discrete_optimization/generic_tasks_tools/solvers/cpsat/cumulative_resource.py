#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Generic

from ortools.sat.python.cp_model import IntervalVar

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.cumulative_resource import (
    CumulativeResourceProblem,
)
from discrete_optimization.generic_tasks_tools.renewable_resource import Resource
from discrete_optimization.generic_tasks_tools.solvers.cpsat.multimode_scheduling import (
    MultimodeSchedulingCpSatSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.renewable_resource import (
    RenewableResourceCpSatSolver,
)


class CumulativeResourceSchedulingCpSatSolver(
    RenewableResourceCpSatSolver[Task, Resource],
    MultimodeSchedulingCpSatSolver[Task],
    Generic[Task, Resource],
):
    """Base class for cpsat solvers dealing with scheduling problems handling cumulative resources."""

    problem: CumulativeResourceProblem[Task, Resource]

    def get_resource_consumption_intervals(
        self, resource: Resource
    ) -> list[tuple[IntervalVar, int]]:
        if self.problem.is_cumulative_resource(resource):
            return [
                (
                    self.get_task_mode_interval(task=task, mode=mode),
                    self.problem.get_renewable_resource_consumption(
                        resource=resource, task=task, mode=mode
                    ),
                )
                for task in self.problem.tasks_list
                for mode in self.problem.get_task_modes(task=task)
            ]
        else:
            raise NotImplementedError(
                f"{resource} is not a cumulative resource whose consumption depends only on task mode."
            )
