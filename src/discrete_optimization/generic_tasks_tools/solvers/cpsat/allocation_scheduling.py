#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from typing import Generic

from ortools.sat.python.cp_model import IntervalVar

from discrete_optimization.generic_tasks_tools.allocation import UnaryResource
from discrete_optimization.generic_tasks_tools.allocation_scheduling import (
    AllocationSchedulingProblem,
    CumulativeResource,
    Resource,
)
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.non_renewable_resource import (
    NonRenewableResource,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.allocation import (
    AllocationCpSatSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.cumulative_resource import (
    CumulativeResourceSchedulingCpSatSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.non_renewable_resource import (
    NonRenewableCpSatSolver,
)


class AllocationSchedulingCpSatSolver(
    CumulativeResourceSchedulingCpSatSolver[Task, Resource],
    NonRenewableCpSatSolver[Task, NonRenewableResource],
    AllocationCpSatSolver[Task, UnaryResource],
    Generic[Task, UnaryResource, CumulativeResource, NonRenewableResource],
):
    problem: AllocationSchedulingProblem[
        Task, UnaryResource, CumulativeResource, NonRenewableResource
    ]

    def get_resource_consumption_intervals(
        self, resource: Resource
    ) -> list[tuple[IntervalVar, int]]:
        if self.problem.is_unary_resource(resource=resource):
            tasks = self.problem.tasks_list
            return [
                (
                    self.get_task_unary_resource_interval(
                        task=task, unary_resource=resource
                    ),
                    1,
                )
                for task in tasks
                if self.problem.is_compatible_task_unary_resource(
                    task=task, unary_resource=resource
                )
            ]
        else:
            return super().get_resource_consumption_intervals(resource=resource)

    @abstractmethod
    def get_task_unary_resource_interval(
        self, task: Task, unary_resource: UnaryResource
    ) -> IntervalVar:
        """Get the interval variable corresponding to given task conditioned to allocation of the given unary resource.

        The method may return an error (no variable existing) if
        `self.problem.is_compatible_task_unary_resource(task=task, unary_resource=unary_resource)` is false.

        """
        ...
