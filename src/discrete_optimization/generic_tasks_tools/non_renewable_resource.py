#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Hashable, Iterable
from typing import Generic, TypeVar

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.multimode import (
    MultimodeProblem,
    MultimodeSolution,
)

logger = logging.getLogger(__name__)

NonRenewableResource = TypeVar("NonRenewableResource", bound=Hashable)


class NonRenewableResourceProblem(
    MultimodeProblem[Task], Generic[Task, NonRenewableResource]
):
    """Base class for problems dealing with non-renewable resources consumed by tasks.

    The task consumption of these non-renewable resources is supposed to be determined entirely determined
    by the task mode.

    """

    @property
    @abstractmethod
    def non_renewable_resources_list(self) -> list[NonRenewableResource]:
        """Non-renewable resources used by the tasks."""
        ...

    @abstractmethod
    def get_non_renewable_resource_capacity(
        self, resource: NonRenewableResource
    ) -> int:
        """Get resource max capacity

        Args:
            resource:

        Returns:

        """
        ...

    @abstractmethod
    def get_non_renewable_resource_consumption(
        self, resource: NonRenewableResource, task: Task, mode: int
    ) -> int:
        """Get resource consumption of the task in the given mode

        Args:
            resource: non-renewable resource
            task:
            mode: not used for single mode problems

        Returns:.

        Raises:
            ValueError: if resource consumption is depending on other variables than mode

        """
        ...


class NonRenewableResourceSolution(
    MultimodeSolution[Task], Generic[Task, NonRenewableResource]
):
    problem: NonRenewableResourceProblem[Task, NonRenewableResource]

    def get_non_renewable_resource_consumption(
        self, resource: NonRenewableResource, task: Task
    ) -> int:
        """Get resource consumption by given task.

        Args:
            resource:
            task:

        Returns:

        """
        return self.problem.get_non_renewable_resource_consumption(
            resource=resource, task=task, mode=self.get_mode(task)
        )

    def check_non_renewable_resource_capacity_constraint(
        self, resource: NonRenewableResource
    ) -> bool:
        """Check capacity constraint on given renewable resource."""
        return self.check_non_renewable_resource_capacity_constraints(
            resources=(resource,)
        )

    def check_non_renewable_resource_capacity_constraints(
        self, resources: Iterable[NonRenewableResource]
    ):
        resources_consumption = {resource: 0 for resource in resources}
        for task in self.problem.tasks_list:
            for resource in resources:
                resources_consumption[resource] += (
                    self.get_non_renewable_resource_consumption(
                        resource=resource, task=task
                    )
                )
        resources_capa_violation = {
            resource: conso
            > self.problem.get_non_renewable_resource_capacity(resource=resource)
            for resource, conso in resources_consumption.items()
        }
        if any(resources_capa_violation.values()):
            logger.debug("Violations on non-renewable resource capacities:")
            for resource, violation in resources_capa_violation.items():
                if violation:
                    logger.debug(f"resource '{resource}'")
            return False
        else:
            return True

    def check_all_non_renewable_resource_capacity_constraints(self) -> bool:
        """Check capacity constraint on all renewable resources."""
        return self.check_non_renewable_resource_capacity_constraints(
            resources=self.problem.non_renewable_resources_list
        )


class WithoutNonRenewableResourceProblem(
    NonRenewableResourceProblem[Task, NonRenewableResource]
):
    """Mixin for problem without non-renewable resources.

    To be used has an additional mixin with generic `AllocationSchedulingProblem`.

    """

    @property
    def non_renewable_resources_list(self) -> list[NonRenewableResource]:
        return []

    def get_non_renewable_resource_capacity(
        self, resource: NonRenewableResource
    ) -> int:
        raise RuntimeError("This problem has no non-renewable resource.")

    def get_non_renewable_resource_consumption(
        self, resource: NonRenewableResource, task: Task, mode: int
    ) -> int:
        raise RuntimeError("This problem has no non-renewable resource.")
