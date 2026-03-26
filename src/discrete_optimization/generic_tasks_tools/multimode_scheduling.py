#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from abc import abstractmethod
from typing import Generic

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.multimode import (
    MultimodeProblem,
    MultimodeSolution,
)
from discrete_optimization.generic_tasks_tools.scheduling import (
    SchedulingProblem,
    SchedulingSolution,
)


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
        return all(
            self.check_task_duration_constraint(task=task)
            for task in self.problem.tasks_list
        )
