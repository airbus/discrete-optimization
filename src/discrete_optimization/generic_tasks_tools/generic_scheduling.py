#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Generic, Union

from discrete_optimization.generic_tasks_tools.allocation import (
    AllocationProblem,
    AllocationSolution,
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.cumulative_resource import (
    CumulativeResource,
    CumulativeResourceProblem,
    CumulativeResourceSolution,
)
from discrete_optimization.generic_tasks_tools.non_renewable_resource import (
    NonRenewableResource,
    NonRenewableResourceProblem,
    NonRenewableResourceSolution,
)
from discrete_optimization.generic_tasks_tools.precedence_scheduling import (
    PrecedenceSchedulingProblem,
    PrecedenceSchedulingSolution,
)

Resource = Union[CumulativeResource, UnaryResource]


class GenericSchedulingProblem(
    CumulativeResourceProblem[Task, CumulativeResource, UnaryResource],
    NonRenewableResourceProblem[Task, NonRenewableResource],
    AllocationProblem[Task, UnaryResource],
    PrecedenceSchedulingProblem[Task],
    Generic[Task, UnaryResource, CumulativeResource, NonRenewableResource],
):
    """Scheduling problem with all optional features

    This class derives from other mixins to provide utilities that require that mix:
    - scheduling: tasks need to be scheduled
    - calendar: the renewable resources have their own calendar that will be used for constraining allocations and schedule
    - multimode: the tasks have several mode on which the duration depends
    - cumulative: the tasks consume cumulative resources according to the chosen mode
    - allocation: the tasks can have unary resources allocated to them
    - non-renewable: the tasks consume non-renewable resources according to the chosen mode
    - precedence: precedence constraints between tasks

    Even though this class is generic but encompasses also more specific cases:
    - singlemode: actually only one mode per task
    - no cumulative ressources: if resources_list list only unary resources
    - no calendar: resource capacity can be given as a constant on [0, horizon)
    - no non-renewable ressources: if non_renewable_resources_list empty
    - no precedence constraints: precedence constraints empty

    We suppose that all renewable resources are
    - either cumulative ones
    - or unary resources

    This generic class is to be used to construct generic automatic solvers (e.g. ).

    """

    @property
    def calendar_resources_list(self) -> list[Resource]:
        return self.unary_resources_list + self.cumulative_resources_list

    def check_calendar_resources_list(self) -> None:
        """Check calendar resources list.

        Raises:
            AssertionError: if duplicates appear in the list

        Returns:

        """
        calendar_resources_list = (
            self.unary_resources_list + self.cumulative_resources_list
        )
        assert len(calendar_resources_list) == len(set(calendar_resources_list)), (
            "There are duplicates in calendar resources list, "
            "potentially because unary and cumulative resources intersect."
        )

    def update_resource_availabilities(self) -> None:
        super().update_resource_availabilities()
        self.check_calendar_resources_list()

    def is_unary_resource(self, resource: Resource) -> bool:
        """Check if given resource is a unary resource."""
        return resource in self.unary_resources_list


class GenericSchedulingSolution(
    CumulativeResourceSolution[Task, CumulativeResource, UnaryResource],
    NonRenewableResourceSolution[Task, NonRenewableResource],
    PrecedenceSchedulingSolution[Task],
    AllocationSolution[Task, UnaryResource],
    Generic[Task, UnaryResource, CumulativeResource, NonRenewableResource],
):
    """Solution type associated to GenericSchedulingProblem."""

    problem: GenericSchedulingProblem[
        Task, UnaryResource, CumulativeResource, NonRenewableResource
    ]

    def get_calendar_resource_consumption(self, resource: Resource, task: Task) -> int:
        """"""
        if self.problem.is_unary_resource(resource=resource):
            # unary resources: 0 (not allocated) or 1 (allocated)
            return int(self.is_allocated(task=task, unary_resource=resource))
        else:
            # cumulative resources
            return super().get_calendar_resource_consumption(
                resource=resource, task=task
            )
