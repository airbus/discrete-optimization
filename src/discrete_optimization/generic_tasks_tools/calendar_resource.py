#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Hashable, Iterable
from typing import Generic, Optional, TypeVar

import numpy as np
import numpy.typing as npt
import wrapt

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.scheduling import (
    SchedulingProblem,
    SchedulingSolution,
)

logger = logging.getLogger(__name__)

Resource = TypeVar("Resource", bound=Hashable)


class CalendarResourceProblem(SchedulingProblem[Task], Generic[Task, Resource]):
    """Base class for scheduling problems dealing with renewable resources whose availability depend on a calendar."""

    @property
    @abstractmethod
    def calendar_resources_list(self) -> list[Resource]:
        """Renewable resources with an availability calendar used by the tasks.

        Notes:
            - renewable = the resource replenishes as soon as a task using it ends;
            - it can be a mix of unary resources (e.g. employees) and cumulative resources (e.g. tool types).
            - calendar can be constant

        """
        ...

    @abstractmethod
    def get_resource_availabilities(
        self, resource: Resource
    ) -> list[tuple[int, int, int]]:
        """Get availabilities intervals for a given resource

        List of availability intervals of a resource.
        If the resource is not available, potentially no interval returned.

        It is assumed that the intervals are disjunct though.

        Args:
            resource:

        Returns:
            list of intervals of the form (start, end, value), which means from time `start` to time `end`,
            there are `value` of the resource available.
            NB: the start is included, the end is excluded (start <= t < end)

        """
        ...

    @wrapt.lru_cache(maxsize=None)
    def get_resource_consolidated_availabilities(
        self, resource: Resource, horizon: Optional[int] = None
    ) -> list[tuple[int, int, int]]:
        """Get availabilities intervals for a given resource, consolidated as a partition of [0, horizon)

        Default implementation use the `get_calendar_resource_availabilities`, by assuming the intervals are disjunct
        but potentially lacking 0-valued intervals.

        Args:
            resource:
            horizon: max value of time considered. Default to `self.get_makespan_upper_bound()`.

        Returns:
            sorted list of intervals of the form (start, end, value), which means from time `start` to time `end`,
            there are `value` of the resource available
            NB: the start is included, the end is excluded (start <= t < end)

        """
        if horizon is None:
            horizon = self.get_makespan_upper_bound()
        return consolidate_availability_intervals(
            intervals=self.get_resource_availabilities(resource=resource),
            horizon=horizon,
        )

    @wrapt.lru_cache(maxsize=None)
    def get_resource_calendar(
        self, resource: Resource, horizon: Optional[int] = None
    ) -> list[int]:
        """Compute resource calendar.

        Args:
            resource:
            horizon: max value of time considered. Default to `self.get_makespan_upper_bound()`.

        Returns:
            list of resource value by time step from 0 to horizon-1.

        """
        if horizon is None:
            horizon = self.get_makespan_upper_bound()
        return convert_availability_intervals_to_calendar(
            intervals=self.get_resource_availabilities(resource=resource),
            horizon=horizon,
        )

    @wrapt.lru_cache(maxsize=None)
    def get_resource_max_capacity(self, resource: Resource) -> int:
        """Get max capacity of the given resource

        Default implementation take the max over its calendar and cache it.

        Args:
            resource:

        Returns:

        """
        return max(
            value
            for start, end, value in self.get_resource_availabilities(resource=resource)
        )

    def update_resource_availabilities(self) -> None:
        """Method to call when the resource availabilities have changed.

        Default implementation clears the cache on `get_resource_max_capacity()`.

        """
        self.get_resource_max_capacity.cache_clear()
        self.get_resource_calendar.cache_clear()
        self.get_resource_consolidated_availabilities.cache_clear()

    def get_fake_tasks(self, resource: Resource) -> list[tuple[int, int, int]]:
        """Get fake tasks explaining the delta between resource current capacity and its max capacity

        Args:
            resource:

        Returns:
            list of intervals of the form (start, end, value),
            for task starting at `start` and ending at `end`, using `value` resource.

        """
        max_capacity = self.get_resource_max_capacity(resource=resource)
        return [
            (start, end, consumption)
            for start, end, value in self.get_resource_consolidated_availabilities(
                resource=resource
            )
            if (consumption := max_capacity - value) > 0
        ]

    def __getstate__(self):
        """Get state for pickle.

        Solve issue when instance has cached methods called. (And thus unpickable cache created.)
        See https://github.com/GrahamDumpleton/wrapt/issues/343

        """
        try:
            dico = super().__getstate__()
        except AttributeError:
            # python < 3.11: __getstate__() not always defined
            dico = self.__dict__
        return {
            key: value
            for key, value in dico.items()
            if not key.startswith("_lru_cache_")
        }


class CalendarResourceSolution(SchedulingSolution[Task], Generic[Task, Resource]):
    problem: CalendarResourceProblem[Task, Resource]

    @abstractmethod
    def get_calendar_resource_consumption(self, resource: Resource, task: Task) -> int:
        """Get resource consumption by given task.

        Args:
            resource:
            task:

        Returns:

        """
        ...

    def check_calendar_resource_capacity_constraint(self, resource: Resource) -> bool:
        """Check capacity constraint on given resource."""
        return self.check_calendar_resource_capacity_constraints(resources=(resource,))

    def _check_calendar_resource_capacity_constraint_naive(
        self, resource: Resource
    ) -> bool:
        makespan = self.get_max_end_time()
        return all(
            sum(
                self.get_calendar_resource_consumption(resource=resource, task=task)
                for task in self.get_running_tasks(time=t)
            )
            <= value
            for t, value in enumerate(
                self.problem.get_resource_calendar(resource=resource)
            )
            if t < makespan
        )

    def _compute_calendar_resource_consumption_np(
        self, resources: Iterable[Resource]
    ) -> dict[Resource, np.ndarray]:
        makespan = self.get_max_end_time()
        resources_consumption = {
            resource: np.zeros(makespan, dtype=int) for resource in resources
        }
        for task in self.problem.tasks_list:
            start = self.get_start_time(task)
            end = self.get_end_time(task)
            for resource in resources:
                resources_consumption[resource][start:end] += (
                    self.get_calendar_resource_consumption(resource=resource, task=task)
                )
        return resources_consumption

    def _check_calendar_resource_capacity_constraint_np(
        self, resources: Iterable[Resource]
    ) -> bool:
        resources_consumption = self._compute_calendar_resource_consumption_np(
            resources=resources
        )
        resources_capa_violation = {
            resource: conso
            > np.array(
                self.problem.get_resource_calendar(
                    resource=resource, horizon=len(conso)
                )
            )
            for resource, conso in resources_consumption.items()
        }
        if any(
            resource_cap_violation.any()
            for resource_cap_violation in resources_capa_violation.values()
        ):
            logger.debug("Violations on calendar resource capacities:")
            violations = {
                resource: violation_timesteps
                for resource, resource_cap_violation in resources_capa_violation.items()
                if len((violation_timesteps := resource_cap_violation.nonzero()[0])) > 0
            }
            for resource, violation_timesteps in violations.items():
                logger.debug(
                    f"resource '{resource}': at time {', '.join(str(t) for t in violation_timesteps)}"
                )
            return False
        else:
            return True

    def check_calendar_resource_capacity_constraints(
        self, resources: Iterable[Resource]
    ) -> bool:
        """Check capacity constraint respected on given resources.

        Do it simultaneously on all resources to optimize computation time.

        """
        return self._check_calendar_resource_capacity_constraint_np(resources=resources)

    def check_all_calendar_resource_capacity_constraints(self) -> bool:
        """Check capacity constraint on all calendar resources."""
        return self.check_calendar_resource_capacity_constraints(
            resources=self.problem.calendar_resources_list
        )

    def compute_calendar_resources_levels(self) -> dict[Resource, int]:
        """Compute the level (i.e. min capacity needed) for each calendar resource."""
        return {
            resource: int(timed_conso.max())
            for resource, timed_conso in self._compute_calendar_resource_consumption_np(
                resources=self.problem.calendar_resources_list
            ).items()
        }

    def compute_aggregated_calendar_resources_levels(
        self, weights: Optional[dict[Resource, int]] = None
    ) -> int:
        """Compute aggregated level (i.e. min capacity needed) of each calendar resource.

        Args:
            weights: optional weights to apply to each resource in the sum. Default to 1.

        """
        if weights is None:
            weights = {}
        return sum(
            conso * weights.get(resource, 1)
            for resource, conso in self.compute_calendar_resources_levels().items()
        )

    def compute_nb_calendar_resources_used(
        self, weights: Optional[dict[Resource, int]] = None
    ) -> int:
        """Compute number of calendar resources used by at least one task.

        Args:
            weights: optional weights to apply to each resource in the sum. Default to 1.


        Returns:

        """
        if weights is None:
            weights = {}
        return sum(
            (conso > 0) * weights.get(resource, 1)
            for resource, conso in self.compute_calendar_resources_levels().items()
        )


NoCalendarResource = None


class WithoutCalendarResourceProblem(
    CalendarResourceProblem[Task, NoCalendarResource], Generic[Task]
):
    @property
    def calendar_resources_list(self) -> list[Resource]:
        return []

    def get_resource_availabilities(
        self, resource: Resource
    ) -> list[tuple[int, int, int]]:
        return []


class WithoutCalendarResourceSolution(
    CalendarResourceSolution[Task, NoCalendarResource], Generic[Task]
):
    def get_calendar_resource_consumption(self, resource: Resource, task: Task) -> int:
        raise ValueError(f"{resource} is not a calendar resource of the problem.")

    def check_calendar_resource_capacity_constraint(self, resource: Resource) -> bool:
        return True

    def check_calendar_resource_capacity_constraints(
        self, resources: Iterable[Resource]
    ) -> bool:
        return True

    def check_all_calendar_resource_capacity_constraints(self) -> bool:
        return True

    def compute_aggregated_calendar_resources_levels(
        self, weights: Optional[dict[Resource, int]] = None
    ):
        return 0

    def compute_nb_calendar_resources_used(
        self, weights: Optional[dict[Resource, int]] = None
    ) -> int:
        return 0

    def compute_calendar_resources_levels(self) -> dict[Resource, int]:
        return {}


def convert_calendar_to_availability_intervals(
    calendar: int | list[int] | npt.NDArray[int], horizon: int
) -> list[tuple[int, int, int]]:
    """Convert a calendar into availability intervals.

    Args:
        calendar: if integer means a constant value, else list of values for each time step.
            If len(calendar)<horizon, last values are assumed to be 0.
            If len(calendar)>horizon, it is truncated to calendar[:horizon]
        horizon: maximum time step considered

    Returns:
        list of (start,end, value), a sorted partition of [0, horizon)

    """
    if np.isscalar(calendar):  # constant resource
        return [(0, horizon, int(calendar))]
    else:  # varying resource
        intervals = []
        t = 0
        start = t
        value = calendar[t]
        end_calendar = min(len(calendar), horizon)
        for t in range(1, end_calendar):
            if calendar[t] != value:
                # ends current interval, starts the next one
                intervals.append((start, t, int(value)))
                start = t
                value = calendar[t]
        intervals.append((start, end_calendar, int(value)))
        return intervals


def consolidate_availability_intervals(
    intervals: list[tuple[int, int, int]], horizon: int
):
    """Ensure that the intervals are a partition of [0, horizon).

    Args:
        intervals: intervals (start, end, value). Supposed to be disjunct.
        horizon: max value of time considered

    Returns:
        sorted list of intervals constituting a partition of [0, horizon)

    """
    # truncate to horizon
    intervals = [
        (start, new_end, value)
        for start, end, value in intervals
        if start < (new_end := min(end, horizon))  # remove empty intervals
    ]

    # sort intervals
    intervals = sorted(
        intervals,
        key=lambda interval: interval[0],
    )

    # look for gaps between intervals => 0 resource
    ends = [0] + [end for start, end, value in intervals]
    next_starts = [start for start, end, value in intervals] + [horizon]
    missing_intervals = []
    for end, next_start in zip(ends, next_starts):
        if end < next_start:
            missing_intervals.append((end, next_start, 0))
        elif end > next_start:
            raise ValueError("Availability intervals intersecting.")
    if len(missing_intervals) > 0:
        intervals = sorted(
            intervals + missing_intervals, key=lambda interval: interval[0]
        )
    return intervals


def convert_availability_intervals_to_calendar(
    intervals: list[tuple[int, int, int]], horizon: int
) -> list[int]:
    """Convert availability intervals into a calendar.

    Args:
        intervals: availability intervals, assumed to be disjunct
        horizon: maximum time step considered

    Returns:
        list of available resource values for each time step

    """
    return [
        value
        for start, end, value in consolidate_availability_intervals(intervals, horizon)
        for _ in range(end - start)
    ]


def merge_resources_availability_intervals(
    intervals_per_resource: list[list[tuple[int, int, int]]], horizon: int
) -> list[tuple[int, int, int]]:
    """Merge several resources availability intervals, considering all resources as one meta-resource

    Args:
        intervals_per_resource: availability for each resource
        horizon: maximum time step considered

    Returns:
        availability intervals for the meta-resource

    """
    return convert_calendar_to_availability_intervals(
        merge_resources_calendars(
            [
                convert_availability_intervals_to_calendar(intervals, horizon=horizon)
                for intervals in intervals_per_resource
            ],
            horizon=horizon,
        ),
        horizon=horizon,
    )


def merge_resources_calendars(calendars: list[list[int]], horizon: int) -> list[int]:
    """Merge several resources calendars, considering all resources as one meta-resource

    Args:
        calendars: calendars for each resource to merge
        horizon: maximum time step considered

    Returns:
        calendar for the meta-resource
    """
    if len(calendars) == 0:
        return [0] * horizon
    else:
        return [
            sum(values)
            for values in zip(
                *(
                    consolidate_calendar(calendar, horizon=horizon)
                    for calendar in calendars
                )
            )
        ]


def consolidate_calendar(calendar: list[int], horizon: int):
    """Consoidate the calendar to correspond to the given horizon


    If too long, retursn calendar[:horizon].
    If too short, fill with 0's.

    Args:
        calendar:
        horizon:

    Returns:
        a calendar of size `horizon`

    """
    return calendar[:horizon] + [0] * (horizon - len(calendar))
