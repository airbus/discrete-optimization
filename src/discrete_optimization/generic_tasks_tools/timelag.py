#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from collections import defaultdict
from typing import Generic

import wrapt

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.enums import MinOrMax, StartOrEnd
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

    def get_original_time_lags(
        self,
        task1_start_or_end: StartOrEnd,
        task2_start_or_end: StartOrEnd,
        min_or_max: MinOrMax,
    ) -> list[tuple[Task, Task, int]]:
        """Maps strt/end/min/max choices to corresponding list of timelags

        Args:
            task1_start_or_end:
            task2_start_or_end:
            min_or_max:

        Returns:

        """
        match task1_start_or_end, task2_start_or_end, min_or_max:
            case StartOrEnd.START, StartOrEnd.START, MinOrMax.MIN:
                timelags = self.get_start_to_start_min_time_lags()
            case StartOrEnd.START, StartOrEnd.START, MinOrMax.MAX:
                timelags = self.get_start_to_start_max_time_lags()
            case StartOrEnd.START, StartOrEnd.END, MinOrMax.MIN:
                timelags = self.get_start_to_end_min_time_lags()
            case StartOrEnd.START, StartOrEnd.END, MinOrMax.MAX:
                timelags = self.get_start_to_end_max_time_lags()
            case StartOrEnd.END, StartOrEnd.START, MinOrMax.MIN:
                timelags = self.get_end_to_start_min_time_lags()
            case StartOrEnd.END, StartOrEnd.START, MinOrMax.MAX:
                timelags = self.get_end_to_start_max_time_lags()
            case StartOrEnd.END, StartOrEnd.END, MinOrMax.MIN:
                timelags = self.get_end_to_end_min_time_lags()
            case StartOrEnd.END, StartOrEnd.END, MinOrMax.MAX:
                timelags = self.get_end_to_end_max_time_lags()
            case _:
                timelags = []
        return timelags

    @wrapt.lru_cache(maxsize=None)
    def get_consolidated_time_lags(
        self,
        task1_start_or_end: StartOrEnd,
        task2_start_or_end: StartOrEnd,
        min_or_max: MinOrMax,
    ) -> list[tuple[Task, Task, int]]:
        """Get consolidated time lags.

        The goal is to normalize the time lags list to avoid having duplicates or constraints implied by others.
        Note that it encompasses
        - same tuples in one of the lists
        - (t1, t2, offset) in end_to_start_min_time_lags and (t2, t1, -offset) in start_to_end_max_time_lags
        - several offsets for the same tasks (t1, t2)

        The choices made here to normalize are:
        1. keep only positive offsets in max time lags (the others are converted to min time lags)
        2. keep only non-negative offsets in min time lags (the others are converted to max time lags)
        3. take max(offsets) in min time lags when several offsets are available for the same tasks (t1, t2)
        4. take min(offsets) in max time lags when several offsets are available for the same tasks (t1, t2)
        5. drop (t1, t2, offset) in max time lag (with offset>0) if (t2, t1, offset') with offset'>=0 is already in corresponding min time lag
          as the latter implies trivially the other

        Note that points 1, 4, and 5 amounts to taking min(offsets) on all tuples in max time lags
        + converted ones from min time lags (t2,t1 -offset), whichever the sign, and drop it if this min(offsets) is non-negative
        (whhich corresponds to the existence of (t2, t1, offset') with offset'>=0 in min time lags.

        Moreover this method makes the mapping between all methods according to start/end/min/max.

        Args:
            task1_start_or_end:
            task2_start_or_end:
            min_or_max:

        Returns:

        """
        if min_or_max == MinOrMax.MAX:
            # max time lags from original + converted min time lags
            timelags = consolidate_max_time_lags(
                self.get_original_time_lags(
                    task1_start_or_end=task1_start_or_end,
                    task2_start_or_end=task2_start_or_end,
                    min_or_max=MinOrMax.MAX,
                )
                + [
                    (t2, t1, -offset)
                    for t1, t2, offset in self.get_original_time_lags(
                        task1_start_or_end=task2_start_or_end,
                        task2_start_or_end=task1_start_or_end,
                        min_or_max=MinOrMax.MIN,
                    )
                ]
            )
            # consolidate by taking min(offsets) for each tuple (t1, t2) + drop it if resulting offset <=0
            # (see docstring for explanation)
            return [
                (t1, t2, offset)
                for t1, t2, offset in consolidate_max_time_lags(timelags)
                if offset > 0
            ]
        else:
            # min time lags with non-negative offsets + converted max time lags with non-positive offsets
            timelags = (
                # original min time lag with
                [
                    (t1, t2, offset)
                    for t1, t2, offset in self.get_original_time_lags(
                        task1_start_or_end=task1_start_or_end,
                        task2_start_or_end=task2_start_or_end,
                        min_or_max=MinOrMax.MIN,
                    )
                    if offset >= 0
                ]
                # corresponding original max time lags tranformed into min time lags
                + [
                    (task2, task1, -offset)
                    for task1, task2, offset in self.get_original_time_lags(
                        task1_start_or_end=task2_start_or_end,
                        task2_start_or_end=task1_start_or_end,
                        min_or_max=MinOrMax.MAX,
                    )
                    if offset <= 0
                ]
            )
            # take max(offsets) for each tuple (t1, t2)
            return consolidate_min_time_lags(timelags=timelags)

    def update_time_lags(self) -> None:
        """Method to call when time lags have been updated.

        Clear cache from consolidated precedence constraints.

        Returns:

        """
        self.get_consolidated_time_lags.cache_clear()

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


def consolidate_min_time_lags(
    timelags: list[tuple[Task, Task, int]],
) -> list[tuple[Task, Task, int]]:
    """Get consolidated min time lags.

    It merges min time lags targeting same tasks, taking the most restrictive offset.

     Args:
         timelags:

     Returns:

    """
    offsets_per_tasks: dict[tuple[Task, Task], set[int]] = defaultdict(set)
    for task1, task2, offset in timelags:
        offsets_per_tasks[task1, task2].add(offset)

    return [
        (task1, task2, max(offsets))
        for (task1, task2), offsets in offsets_per_tasks.items()
    ]


def consolidate_max_time_lags(
    timelags: list[tuple[Task, Task, int]],
) -> list[tuple[Task, Task, int]]:
    """Get consolidated min time lags.

    It merges min time lags targeting same tasks, taking the most restrictive offset.

     Args:
         timelags:

     Returns:

    """
    offsets_per_tasks: dict[tuple[Task, Task], set[int]] = defaultdict(set)
    for task1, task2, offset in timelags:
        offsets_per_tasks[task1, task2].add(offset)

    return [
        (task1, task2, min(offsets))
        for (task1, task2), offsets in offsets_per_tasks.items()
    ]


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
