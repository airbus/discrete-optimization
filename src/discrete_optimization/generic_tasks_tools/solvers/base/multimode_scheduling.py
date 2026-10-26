#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from typing import Generic

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.multimode_scheduling import (
    MultimodeSchedulingProblem,
    SinglemodeSchedulingProblem,
)
from discrete_optimization.generic_tasks_tools.solvers.base.multimode import (
    MultimodeCpBaseSolver,
    SinglemodeCpBaseSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.base.scheduling import (
    SchedulingCpBaseSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.base.typing import IntervalVar


class MultimodeSchedulingCpBaseSolver(
    SchedulingCpBaseSolver[Task], MultimodeCpBaseSolver[Task], Generic[Task]
):
    """Base class for solvers dealing with scheduling problems whose tasks durations depend only on mode."""

    problem: MultimodeSchedulingProblem[Task]

    @abstractmethod
    def get_task_mode_interval(self, task: Task, mode: int) -> IntervalVar:
        """Get the interval variable corresponding to given task and mode.

        Needs to be created if avoid_interval_optional_for_cumulative_resources is False.
        Not necessary else.

        The interval should be optional with the is_present variable being false
        whenever the task is optional and absent.

        """
        ...


class SinglemodeSchedulingCpBaseSolver(
    MultimodeSchedulingCpBaseSolver[Task], SinglemodeCpBaseSolver[Task]
):
    """Base class for solvers dealing with single mode scheduling problems with fixed tasks durations."""

    problem: SinglemodeSchedulingProblem[Task]

    @abstractmethod
    def get_task_interval(self, task: Task) -> IntervalVar:
        """Get the interval variable corresponding to given task."""
        ...

    def get_task_mode_interval(self, task: Task, mode: int) -> IntervalVar:
        return self.get_task_interval(task=task)
