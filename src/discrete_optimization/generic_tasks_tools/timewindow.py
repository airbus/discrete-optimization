#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from typing import Generic

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.enums import MinOrMax, StartOrEnd
from discrete_optimization.generic_tasks_tools.scheduling import (
    SchedulingProblem,
    SchedulingSolution,
)

logger = logging.getLogger(__name__)


class TimewindowProblem(SchedulingProblem[Task], Generic[Task]):
    """Class for problem having time windows between tasks."""

    def get_task_bound(
        self, task: Task, start_or_end: StartOrEnd, min_or_max: MinOrMax
    ) -> int:
        """Get a lower or upper bound on a task start or end.

        Args:
            task:
            start_or_end:
            min_or_max: min -> lower bound, max -> upper bound

        Returns:

        """
        if min_or_max == MinOrMax.MIN:
            return self.get_task_start_or_end_lower_bound(
                task=task, start_or_end=start_or_end
            )
        else:
            return self.get_task_start_or_end_upper_bound(
                task=task, start_or_end=start_or_end
            )

    def get_task_start_or_end_lower_bound(
        self, task: Task, start_or_end: StartOrEnd
    ) -> int:
        """Get a lower bound on start or end of a given task as specified by the problem.

        For tighter computed bounds, see `GenericSchedulingProblem.get_tight_task_start_or_end_lower_bound()` and
        `GenericSchedulingProblem.compute_task_bounds()`.

        Default implementation: 0

        Args:
            task:
            start_or_end:

        Returns:

        """
        return 0

    def get_task_start_or_end_upper_bound(
        self, task: Task, start_or_end: StartOrEnd
    ) -> int:
        """Get an upper bound on start or end of a given task as specified by the problem.

        For tighter computed bounds, see `GenericSchedulingProblem.get_tight_task_start_or_end_upper_bound()` and
        `GenericSchedulingProblem.compute_task_bounds()`.

        Default implementation: makespan upper bound

        Args:
            task:
            start_or_end:

        Returns:

        """
        return self.get_makespan_upper_bound()

    def get_makespan_lower_bound(self) -> int:
        """Get a lower bound on global makespan.

        Time windows on last tasks can be used to get a better makespan lower bound.

        """
        return max(
            self.get_task_start_or_end_lower_bound(
                task=task, start_or_end=StartOrEnd.END
            )
            for task in self.get_last_tasks()
        )


class TimewindowSolution(SchedulingSolution[Task], Generic[Task]):
    """Class for solution of problems having time windows between tasks."""

    problem: TimewindowProblem[Task]

    def check_time_windows(self) -> bool:
        """check whether time windows are respected."""
        for task in self.problem.tasks_list:
            start = self.get_start_time(task)
            if start < self.problem.get_task_start_or_end_lower_bound(
                task=task, start_or_end=StartOrEnd.START
            ) or start > self.problem.get_task_start_or_end_upper_bound(
                task=task, start_or_end=StartOrEnd.START
            ):
                logger.debug(f"Window for start of {task} is not respected.")
                return False
            end = self.get_end_time(task)
            if end < self.problem.get_task_start_or_end_lower_bound(
                task=task, start_or_end=StartOrEnd.END
            ) or end > self.problem.get_task_start_or_end_upper_bound(
                task=task, start_or_end=StartOrEnd.END
            ):
                logger.debug(f"Window for end of {task} is not respected.")
                return False

        return True
