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
        """Map each task to the tasks that need to be performed after it."""
        ...


class NoOverlapSolution(SchedulingSolution[Task]):
    """Solution for problem with precedence constraints."""

    problem: NoOverlapProblem[Task]

    def check_no_overlap(self):
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
                    logger.info(f"{task} has overlap inside {tasks}")
                    return False
                cumul_use[(st - min_start) : (end - min_start)] += 1
        return True


class WithoutNoOverlapProblem(NoOverlapProblem[Task]):
    """Utility mixin for problem w/o precedence constraints.

    To be used has an additional mixin with generic `GenericSchedulingProblem`.

    """

    def get_no_overlap(self) -> set[frozenset[Task]]:
        return set()


class WithoutNoOverlapSolution(NoOverlapSolution[Task]):
    def check_no_overlap(self) -> bool:
        return True
