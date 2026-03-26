#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

from abc import abstractmethod
from typing import Generic

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.multimode_scheduling import (
    MultimodeSchedulingProblem,
    MultimodeSchedulingSolution,
)
from discrete_optimization.generic_tasks_tools.renewable_resource import (
    RenewableResourceProblem,
    RenewableResourceSolution,
    Resource,
)


class CumulativeResourceProblem(
    RenewableResourceProblem[Task, Resource],
    MultimodeSchedulingProblem[Task],
    Generic[Task, Resource],
):
    """Scheduling problem with cumulative resources consumed by task.

    This derives from problem with renewable resources, some of them are cumulative, some are not (e.g. unary resource
    if it is moreover an allocation problem).
    The task consumption of these cumulative resources is supposed to be determined entirely determined by the task mode.

    """

    @abstractmethod
    def get_resource_consumption(
        self, resource: Resource, task: Task, mode: int
    ) -> int:
        """Get resource consumption of the task in the given mode

        Args:
            resource:
            task:
            mode: not used for single mode problems

        Returns:
            the consumption for cumulative resources.

        Raises:
            ValueError: if resource consumption is depending on other variables than mode

        """
        ...

    @abstractmethod
    def is_cumulative_resource(self, resource: Resource) -> bool:
        """Check if given resource is a cumulative resource whose consumption depends only on task mode.

        Args:
            resource:

        Returns:

        """
        ...


class CumulativeResourceSolution(
    RenewableResourceSolution[Task, Resource],
    MultimodeSchedulingSolution[Task],
    Generic[Task, Resource],
):
    """Solution type associated to CumulativeResourceProblem."""

    problem: CumulativeResourceProblem[Task, Resource]

    def get_resource_consumption(self, resource: Resource, task: Task) -> int:
        """Get resource consumption by given task.

        Default implementation works only for cumulative resources whose consumptions depend only on task mode.

        Args:
            resource:
            task:

        Returns:

        """
        if self.problem.is_cumulative_resource(resource):
            return self.problem.get_resource_consumption(
                resource=resource, task=task, mode=self.get_mode(task)
            )
        else:
            raise NotImplementedError(
                f"{resource} is not a cumulative resource whose consumption depends only on task mode."
            )
