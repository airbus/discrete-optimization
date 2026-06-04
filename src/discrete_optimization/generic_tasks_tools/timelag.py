#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from typing import Generic

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.scheduling import (
    SchedulingProblem,
    SchedulingSolution,
)

logger = logging.getLogger(__name__)


class TimelagProblem(SchedulingProblem[Task], Generic[Task]):
    """Class for problem having time lags between tasks."""

    def get_start_to_start_min_time_lags(self) -> list[tuple[Task, Task, int]]:
        """Get min time lags between tasks starts.

        Default to no min time lags. Should be overriden in child class for problems with min time lags.


        Returns:
            list of task1, task2, offset meaning start(task1) + offset <= start(task2)

        """
        return []

    def get_start_to_start_max_time_lags(self) -> list[tuple[Task, Task, int]]:
        """Get max time lags between tasks starts.

        Default to no max time lags. Should be overriden in child class for problems with max time lags.


        Returns:
            list of task1, task2, offset meaning start(task1) + offset >= start(task2)

        """
        return []

    def get_end_to_start_min_time_lags(self) -> list[tuple[Task, Task, int]]:
        """Get min time lags between first task end and second task start.

        Default to no min time lags. Should be overriden in child class for problems with min time lags.


        Returns:
            list of task1, task2, offset meaning end(task1) + offset <= start(task2)

        """
        return []

    def get_end_to_start_max_time_lags(self) -> list[tuple[Task, Task, int]]:
        """Get max time lags between first task end and second task start.

        Default to no max time lags. Should be overriden in child class for problems with max time lags.


        Returns:
            list of task1, task2, offset meaning end(task1) + offset >= start(task2)

        """
        return []

    def get_end_to_end_min_time_lags(self) -> list[tuple[Task, Task, int]]:
        """Get min time lags between task ends.

        Default to no min time lags. Should be overriden in child class for problems with min time lags.


        Returns:
            list of task1, task2, offset meaning end(task1) + offset <= end(task2)

        """
        return []

    def get_end_to_end_max_time_lags(self) -> list[tuple[Task, Task, int]]:
        """Get max time lags between task ends.

        Default to no max time lags. Should be overriden in child class for problems with max time lags.


        Returns:
            list of task1, task2, offset meaning end(task1) + offset >= end(task2)

        """
        return []

    def get_start_to_end_min_time_lags(self) -> list[tuple[Task, Task, int]]:
        """Get min time lags between first task start and second task end.

        Default to no min time lags. Should be overriden in child class for problems with min time lags.


        Returns:
            list of task1, task2, offset meaning start(task1) + offset <= end(task2)

        """
        return []

    def get_start_to_end_max_time_lags(self) -> list[tuple[Task, Task, int]]:
        """Get max time lags between first task start and second task end.

        Default to no max time lags. Should be overriden in child class for problems with max time lags.


        Returns:
            list of task1, task2, offset meaning start(task1) + offset >= end(task2)

        """
        return []


class TimelagSolution(SchedulingSolution[Task], Generic[Task]):
    """Class for solution of problems having time lags between tasks."""

    problem: TimelagProblem[Task]

    def check_time_lags(self) -> bool:
        """check whether time lags are respected."""
        for task1, task2, offset in self.problem.get_start_to_start_min_time_lags():
            if self.get_start_time(task2) < self.get_start_time(task1) + offset:
                logger.debug(
                    f"Min time lag ({offset}) not respected between {task1} start and {task2} start."
                )
                return False
        for task1, task2, offset in self.problem.get_start_to_start_max_time_lags():
            if self.get_start_time(task2) > self.get_start_time(task1) + offset:
                logger.debug(
                    f"Max time lag ({offset}) not respected between {task1} start and {task2} start."
                )
                return False
        for task1, task2, offset in self.problem.get_end_to_start_min_time_lags():
            if self.get_start_time(task2) < self.get_end_time(task1) + offset:
                logger.debug(
                    f"Min time lag ({offset}) not respected between {task1} end and {task2} start."
                )
                return False
        for task1, task2, offset in self.problem.get_end_to_start_max_time_lags():
            if self.get_start_time(task2) > self.get_end_time(task1) + offset:
                logger.debug(
                    f"Max time lag ({offset}) not respected between {task1} end and {task2} start."
                )
                return False
        for task1, task2, offset in self.problem.get_end_to_end_min_time_lags():
            if self.get_end_time(task2) < self.get_end_time(task1) + offset:
                logger.debug(
                    f"Min time lag ({offset}) not respected between {task1} end and {task2} end."
                )
                return False
        for task1, task2, offset in self.problem.get_end_to_end_max_time_lags():
            if self.get_end_time(task2) > self.get_end_time(task1) + offset:
                logger.debug(
                    f"Max time lag ({offset}) not respected between {task1} end and {task2} end."
                )
                return False
        for task1, task2, offset in self.problem.get_start_to_end_min_time_lags():
            if self.get_end_time(task2) < self.get_start_time(task1) + offset:
                logger.debug(
                    f"Min time lag ({offset}) not respected between {task1} start and {task2} end."
                )
                return False
        for task1, task2, offset in self.problem.get_start_to_end_max_time_lags():
            if self.get_end_time(task2) > self.get_start_time(task1) + offset:
                logger.debug(
                    f"Max time lag ({offset}) not respected between {task1} start and {task2} end."
                )
                return False
        return True
