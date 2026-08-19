#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

import logging
from abc import abstractmethod
from functools import reduce
from typing import Generic, Hashable, TypeVar

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.calendar_resource import (
    CalendarResourceProblem,
    CalendarResourceSolution,
)
from discrete_optimization.generic_tasks_tools.multimode_scheduling import (
    MultimodeSchedulingProblem,
    MultimodeSchedulingSolution,
)

CumulativeResource = TypeVar("CumulativeResource", bound=Hashable)
OtherCalendarResource = TypeVar("OtherCalendarResource", bound=Hashable)
Resource = CumulativeResource | OtherCalendarResource


class CumulativeResourceProblem(
    CalendarResourceProblem[Task, Resource],
    MultimodeSchedulingProblem[Task],
    Generic[Task, CumulativeResource, OtherCalendarResource],
):
    """Scheduling problem with cumulative resources consumed by task.

    This derives from problem with renewable calendar resources, some of them are cumulative, some are not (e.g. unary resource
    if it is moreover an allocation problem).
    The task consumption of these cumulative resources is supposed to be determined entirely determined by the task mode.

    """

    def get_cumulative_resource_consumption_mapping(
        self, resource: CumulativeResource, task: Task, mode: int
    ) -> dict[frozenset[tuple[Task, int]], int]:
        # To be Overridden in child classes
        return {}

    def get_possible_resource_consumption(
        self, resource: CumulativeResource, task: Task, mode: int
    ) -> set[int]:
        if self.is_resource_task_mode_consumption_dependent(
            resource=resource, task=task, mode=mode
        ):
            return set(
                self.get_cumulative_resource_consumption_mapping(
                    resource=resource, task=task, mode=mode
                ).values()
            )
        return {
            self.get_cumulative_resource_consumption(
                resource=resource, task=task, mode=mode
            )
        }

    def get_possible_resource_consumption_all_modes(
        self, resource: CumulativeResource, task: Task
    ) -> set[int]:
        return reduce(
            lambda prev, y: prev.union(
                self.get_possible_resource_consumption(
                    resource=resource, task=task, mode=y
                )
            ),
            list(self.get_task_modes(task)),
            set(),
        )

    def is_resource_task_mode_consumption_dependent(
        self, resource: CumulativeResource, task: Task, mode: int
    ) -> bool:
        # To be Overridden in child classes
        return False

    def is_resource_task_consumption_dependent(
        self, resource: CumulativeResource, task: Task
    ):
        return any(
            self.is_resource_task_mode_consumption_dependent(
                resource=resource, task=task, mode=mode
            )
            for mode in self.get_task_modes(task)
        )

    def is_task_consumption_dependent(self, task: Task):
        # To be Overridden in child classes
        return any(
            self.is_resource_task_consumption_dependent(resource=resource, task=task)
            for resource in self.cumulative_resources_list
        )

    def has_any_consumption_dependent(self):
        return any(self.is_task_consumption_dependent(task) for task in self.tasks_list)

    @abstractmethod
    def get_cumulative_resource_consumption(
        self, resource: CumulativeResource, task: Task, mode: int
    ) -> int:
        """Get cumulative resource consumption of the task in the given mode

        Args:
            resource: cumulative resource
            task:
            mode: not used for single mode problems

        Returns:
            the consumption for cumulative resources.

        """
        ...

    @property
    @abstractmethod
    def cumulative_resources_list(self) -> list[CumulativeResource]: ...

    def is_cumulative_resource(self, resource: Resource) -> bool:
        """Check if given resource is a cumulative resource whose consumption depends only on task mode.

        Args:
            resource:

        Returns:

        """
        return resource in self.cumulative_resources_list


class CumulativeResourceSolution(
    CalendarResourceSolution[Task, Resource],
    MultimodeSchedulingSolution[Task],
    Generic[Task, CumulativeResource, OtherCalendarResource],
):
    """Solution type associated to CumulativeResourceProblem."""

    problem: CumulativeResourceProblem[Task, CumulativeResource, OtherCalendarResource]

    def get_calendar_resource_consumption_from_mapping(
        self, resource: CumulativeResource, task: Task
    ) -> int:
        mode = self.get_mode(task)
        mapping = self.problem.get_cumulative_resource_consumption_mapping(
            resource=resource, task=task, mode=mode
        )
        set_of_tasks = set([frozenset([t for t, m in k]) for k in mapping])
        value = next(
            (
                mapping[key_mapping]
                for set_task in set_of_tasks
                if (key_mapping := frozenset([(t, self.get_mode(t)) for t in set_task]))
                in mapping
            ),
            None,
        )
        if value is None:
            logging.info(f"No found mapping")
            return None
        return value

    def get_calendar_resource_consumption(self, resource: Resource, task: Task) -> int:
        """Get resource consumption by given task.

        Default implementation works only for cumulative resources whose consumptions depend only on task mode.

        Args:
            resource:
            task:

        Returns:

        """
        if self.problem.is_cumulative_resource(resource):
            if not self.problem.is_resource_task_mode_consumption_dependent(
                resource, task, self.get_mode(task)
            ):
                return self.problem.get_cumulative_resource_consumption(
                    resource=resource, task=task, mode=self.get_mode(task)
                )
            else:
                return self.get_calendar_resource_consumption_from_mapping(
                    resource=resource, task=task
                )

        else:
            raise NotImplementedError(
                f"{resource} is not a cumulative resource whose consumption depends only on task mode."
            )


NoCumulativeResource = None


class WithoutCumulativeResourceProblem(
    CumulativeResourceProblem[Task, NoCumulativeResource, OtherCalendarResource],
    Generic[Task, OtherCalendarResource],
):
    """Mixin for problem without cumulative resources.

    To be used has an additional mixin with generic `GenericSchedulingProblem`.

    """

    @property
    def cumulative_resources_list(self) -> list[CumulativeResource]:
        return []

    def get_cumulative_resource_consumption(
        self, resource: CumulativeResource, task: Task, mode: int
    ) -> int:
        raise ValueError(f"{resource} is not a cumulative resource of the problem.")


class WithoutCumulativeResourceSolution(
    CumulativeResourceSolution[Task, NoCumulativeResource, OtherCalendarResource],
    Generic[Task, OtherCalendarResource],
):
    """Mixin for solution without cumulative resources.

    To be used has an additional mixin with generic `GenericSchedulingSolution`.

    """

    ...
