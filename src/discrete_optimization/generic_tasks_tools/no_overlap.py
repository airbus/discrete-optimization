from __future__ import annotations

import logging
from abc import abstractmethod

import numpy as np

from discrete_optimization.generic_tasks_tools.scheduling import (
    SchedulingProblem,
    SchedulingSolution,
    Task,
)

logger = logging.getLogger(__name__)


class NoOverlapProblem(SchedulingProblem[Task]):
    """Problem with no overlap between tasks:
    For example for open-shop problem, where the order of each operations
    is arbitrary, but still no overlap.
    """

    @abstractmethod
    def get_no_overlap(self) -> set[frozenset[Task]]:
        """
        An object in this returned set is a (frozen) set of task,
        where no task should overlap with another one in this set
        """
        ...

    def get_forbidden_intervals(self, task: Task) -> list[tuple[int, int]]:
        """Get fixed intervals that should not overlap with given task.

        Default to empty list. To be overridden in child classes.

        Args:
            task:

        Returns:
            List of intervals (start, end) with start <= end (will not be checked)

        """
        return []


class NoOverlapSolution(SchedulingSolution[Task]):
    """Solution for problem with precedence constraints."""

    problem: NoOverlapProblem[Task]

    def check_no_overlap(self) -> bool:
        # Doesnt work perfectly for zero duration tasks,
        # this is more equivalent to cumulative with capacity 1.
        for tasks in self.problem.get_no_overlap():
            min_start = min([self.get_start_time(t) for t in tasks])
            max_end = max([self.get_end_time(t) for t in tasks])
            cumul_use = np.zeros((max_end - min_start))
            for task in tasks:
                st, end = self.get_start_time(task), self.get_end_time(task)
                if end == st:
                    continue
                if np.max(cumul_use[(st - min_start) : (end - min_start)]) == 1:
                    logger.debug(f"Task {task} has overlap inside {tasks}")
                    return False
                cumul_use[(st - min_start) : (end - min_start)] += 1
        return True

    def check_forbidden_intervals(self) -> bool:
        for task in self.problem.tasks_list:
            intervals = self.problem.get_forbidden_intervals(task)
            if len(intervals) > 0:
                start1 = self.get_start_time(task)
                end1 = self.get_end_time(task)
                if any(
                    _check_intervals_intersect(start1, end1, start2, end2)
                    for start2, end2 in intervals
                ):
                    logger.debug(f"Task {task} overlaps with its forbidden intervals.")
                    return False
        return True


def _check_intervals_intersect(start1: int, end1: int, start2: int, end2: int) -> bool:
    """Check whether two intervals are intersecting

    We assume that start1<end1 and start2<end2.

    Args:
        start1: start of first interval
        end1: end of first interval
        start2: start of second interval
        end2: end of second interval

    Returns:

    """
    return not ((end1 <= start2) or (end2 <= start1))


class WithoutNoOverlapProblem(NoOverlapProblem[Task]):
    """Utility mixin for problem w/o precedence constraints.

    To be used has an additional mixin with generic `GenericSchedulingProblem`.

    """

    def get_no_overlap(self) -> set[frozenset[Task]]:
        return set()


class WithoutNoOverlapSolution(NoOverlapSolution[Task]):
    def check_no_overlap(self) -> bool:
        return True
