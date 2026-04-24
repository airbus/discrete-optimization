#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from abc import abstractmethod
from collections.abc import Hashable
from typing import Generic, TypeVar, Union

from discrete_optimization.generic_tasks_tools.allocation import (
    AllocationProblem,
    AllocationSolution,
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.cumulative_resource import (
    CumulativeResourceProblem,
    CumulativeResourceSolution,
)
from discrete_optimization.generic_tasks_tools.non_renewable_resource import (
    NonRenewableResource,
    NonRenewableResourceProblem,
    NonRenewableResourceSolution,
)

CumulativeResource = TypeVar("CumulativeResource", bound=Hashable)
Resource = Union[CumulativeResource, UnaryResource]


class AllocationSchedulingProblem(
    CumulativeResourceProblem[Task, Resource],
    NonRenewableResourceProblem[Task, NonRenewableResource],
    AllocationProblem[Task, UnaryResource],
    Generic[Task, UnaryResource, CumulativeResource, NonRenewableResource],
):
    """Scheduling problem with unary resource allocation.

    This class derives from other mixins to provide utilities that require that mix:
    - renewable: the unary resources have their own calendar that will be used for constraining allocations
    - multimode: the tasks have several mode on which the duration depends
    - cumulative: the tasks consume cumulative resources according to the chosen mode
    - allocation
    - non-renewable: the tasks consume non-renewable resources according to the chosen mode

    Even though this class is generic but encompasses also more specific cases:
    - singlemode: actually only one mode per task
    - no cumulative ressources: if resources_list list only unary resources
    - no calendar: resource capacity can be given as a constant on [0, horizon)
    - no non-renewable ressources: if non_renewable_resources_list empty

    We suppose that all renewable resources are
    - either cumulative ones
    - or unary resources

    This generic class is to be used to construct generic solvers (e.g. cpsat)
    that will require no methods implementation to work.

    """

    @property
    @abstractmethod
    def cumulative_resources_list(self) -> list[CumulativeResource]: ...

    @property
    def renewable_resources_list(self) -> list[Resource]:
        return self.unary_resources_list + self.cumulative_resources_list

    def check_renewable_resources_list(self) -> None:
        """Check renewable resources list.

        Raises:
            AssertionError: if duplicates appear in the list

        Returns:

        """
        renewable_resources_list = (
            self.unary_resources_list + self.cumulative_resources_list
        )
        assert len(renewable_resources_list) == len(set(renewable_resources_list)), (
            "There are duplicates in renewable resources list, "
            "potentially because unary and cumulative resources intersect."
        )

    def update_resource_availabilities(self) -> None:
        super().update_resource_availabilities()
        self.check_renewable_resources_list()

    def is_cumulative_resource(self, resource: Resource) -> bool:
        """Check if given resource is a cumulative resource."""
        return resource in self.cumulative_resources_list

    def is_unary_resource(self, resource: Resource) -> bool:
        """Check if given resource is a unary resource."""
        return resource in self.unary_resources_list


class AllocationSchedulingSolution(
    CumulativeResourceSolution[Task, Resource],
    NonRenewableResourceSolution[Task, NonRenewableResource],
    AllocationSolution[Task, UnaryResource],
    Generic[Task, UnaryResource, CumulativeResource, NonRenewableResource],
):
    """Solution type associated to AllocationSchedulingProblem."""

    problem: AllocationSchedulingProblem[
        Task, UnaryResource, CumulativeResource, NonRenewableResource
    ]

    def get_renewable_resource_consumption(self, resource: Resource, task: Task) -> int:
        """"""
        if self.problem.is_unary_resource(resource=resource):
            # unary resources: 0 (not allocated) or 1 (allocated)
            return int(self.is_allocated(task=task, unary_resource=resource))
        else:
            # cumulative resources
            return super().get_renewable_resource_consumption(
                resource=resource, task=task
            )
