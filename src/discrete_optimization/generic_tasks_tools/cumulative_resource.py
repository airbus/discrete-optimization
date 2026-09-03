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
    """Scheduling problem with cumulative resources consumed by tasks.

    Supports two consumption modes:

    1. **Standard**: Task consumption is fixed by task mode.
       Example: Task A in mode 1 always consumes 5 units.

    2. **Resource-dependent**: Task consumption depends on other tasks' modes.
       Modeled via a consumption mapping.

    **Resource-Dependent Example:**

    Task A's electricity varies based on whether task B's heater is on:

        >>> mapping = {
        ...     frozenset([(task_B, 0)]): 10,  # B heater off: A uses 10 kW
        ...     frozenset([(task_B, 1)]): 8,   # B heater on: A uses 8 kW
        ... }

    Implementation pattern:

        >>> class MyProblem(CumulativeResourceProblem):
        ...     def is_cumulative_resource_task_mode_consumption_dependent(
        ...         self, resource, task, mode
        ...     ):
        ...         return resource in self.mode_details[task][mode]
        ...
        ...     def get_cumulative_resource_consumption_mapping(
        ...         self, resource, task, mode
        ...     ):
        ...         if self.is_cumulative_resource_task_mode_consumption_dependent(
        ...             resource, task, mode
        ...         ):
        ...             return self.mode_details[task][mode][resource]
        ...         return {frozenset([]): self.get_cumulative_resource_consumption(
        ...             resource, task, mode
        ...         )}

    See `src/discrete_optimization/rcpsp_resource_dependent/problem.py` for a complete implementation.

    """

    def get_cumulative_resource_consumption_mapping(
        self, resource: CumulativeResource, task: Task, mode: int
    ) -> dict[frozenset[tuple[Task, int]], int]:
        """Get resource consumption mapping for resource-dependent tasks.

        Args:
            resource: The cumulative resource
            task: The task
            mode: The task mode

        Returns:
            Mapping from task/mode configurations to consumption values.
            Keys are frozensets of (task, mode) tuples.
            Returns {} if task has standard (non-dependent) consumption.

        Example:
            >>> mapping = problem.get_cumulative_resource_consumption_mapping(
            ...     "electricity", task_A, 0
            ... )
            >>> # {frozenset([(task_B, 0)]): 100, frozenset([(task_B, 1)]): 80}
        """
        # To be overridden in child classes
        return {}

    def get_possible_cumulative_resource_consumption(
        self, resource: CumulativeResource, task: Task, mode: int
    ) -> set[int]:
        if self.is_cumulative_resource_task_mode_consumption_dependent(
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

    def get_possible_cumulative_resource_consumption_all_modes(
        self, resource: CumulativeResource, task: Task
    ) -> set[int]:
        return reduce(
            lambda prev, y: prev.union(
                self.get_possible_cumulative_resource_consumption(
                    resource=resource, task=task, mode=y
                )
            ),
            list(self.get_task_modes(task)),
            set(),
        )

    def is_cumulative_resource_task_mode_consumption_dependent(
        self, resource: CumulativeResource, task: Task, mode: int
    ) -> bool:
        # To be Overridden in child classes
        return False

    def is_cumulative_resource_task_consumption_dependent(
        self, resource: CumulativeResource, task: Task
    ):
        return any(
            self.is_cumulative_resource_task_mode_consumption_dependent(
                resource=resource, task=task, mode=mode
            )
            for mode in self.get_task_modes(task)
        )

    def is_task_cumulative_consumption_dependent(self, task: Task):
        # To be Overridden in child classes
        return any(
            self.is_cumulative_resource_task_consumption_dependent(
                resource=resource, task=task
            )
            for resource in self.cumulative_resources_list
        )

    def has_any_cumulative_consumption_dependent(self):
        return any(
            self.is_task_cumulative_consumption_dependent(task)
            for task in self.tasks_list
        )

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
        """Retrieve resource consumption from mapping based on current solution's mode assignments.

        Args:
            resource: The cumulative resource
            task: The task

        Returns:
            Resource consumption value, or None if no matching configuration found.

        Example:
            Given mapping {frozenset([(task_B, 1)]): 8, frozenset([(task_B, 0)]): 10}
            and solution with task_B in mode 1, returns 8.
        """
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

        Dispatches to either:
        - `get_cumulative_resource_consumption()` for standard (fixed) consumption
        - `get_calendar_resource_consumption_from_mapping()` for resource-dependent consumption

        Args:
            resource: The calendar resource (must be cumulative)
            task: The task

        Returns:
            Resource consumption amount

        Raises:
            NotImplementedError: If resource is not cumulative
        """
        if self.problem.is_cumulative_resource(resource):
            if not self.problem.is_cumulative_resource_task_mode_consumption_dependent(
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
