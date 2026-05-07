#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from abc import abstractmethod
from typing import Generic

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.multimode import (
    MultimodeProblem,
    MultimodeSolution,
    SinglemodeProblem,
    SinglemodeSolution,
)
from discrete_optimization.generic_tasks_tools.scheduling import (
    SchedulingProblem,
    SchedulingSolution,
)

logger = logging.getLogger(__name__)


class MultimodeSchedulingProblem(
    SchedulingProblem[Task],
    MultimodeProblem[Task],
    Generic[Task],
):
    """Scheduling problem whose tasks durations depend only on mode.

    This derives from problem with renewable resources, some of them are cumulative, some are not (e.g. unary resource
    if it is moreover an allocation problem).
    The task consumption of these cumulative resources is supposed to be determined entirely determined by the task mode.

    """

    @abstractmethod
    def get_task_mode_duration(self, task: Task, mode: int) -> int:
        """Get task duration according to mode.

        Args:
            task:
            mode: not used for single-mode problems

        Returns:

        """
        ...


class SinglemodeSchedulingProblem(
    SinglemodeProblem[Task],
    MultimodeSchedulingProblem[Task],
):
    """Single mode scheduling problems with fixed task durations.

    Utility class simplifying MultimodeSchedulingProblem when single mode only.

    """

    @abstractmethod
    def get_task_duration(self, task: Task) -> int:
        """Get task duration according to mode.

        Args:
            task:

        Returns:

        """
        ...

    def get_task_mode_duration(self, task: Task, mode: int) -> int:
        return self.get_task_duration(task=task)


class MultimodeSchedulingSolution(
    SchedulingSolution[Task],
    MultimodeSolution[Task],
    Generic[Task],
):
    """Solution type associated to MultimodeSchedulingProblem."""

    problem: MultimodeSchedulingProblem[Task]

    def check_task_duration_constraint(self, task: Task) -> bool:
        return self.problem.get_task_mode_duration(
            task=task, mode=self.get_mode(task=task)
        ) == self.get_duration(task=task)

    def check_duration_constraints(self) -> bool:
        check = all(
            self.check_task_duration_constraint(task=task)
            for task in self.problem.tasks_list
        )
        if not check:
            for task in self.problem.tasks_list:
                if not self.check_task_duration_constraint(task=task):
                    logger.debug(
                        f"Duration of task {task} not consistent with the mode choice."
                    )
                    break
        return check


class SinglemodeSchedulingSolution(
    SinglemodeSolution[Task],
    MultimodeSchedulingSolution[Task],
):
    """Solution for single mode scheduling problem with fixed task durations.

    Utility class useful when needing to derive from GenericSchedulingSolution without multi mode
    to be able to use cpsat auto solver.

    """
