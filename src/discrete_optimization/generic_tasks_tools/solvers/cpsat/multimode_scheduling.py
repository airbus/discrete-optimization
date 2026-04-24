#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from typing import Generic

from ortools.sat.python.cp_model import IntervalVar

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.multimode_scheduling import (
    MultimodeSchedulingProblem,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.multimode import (
    MultimodeCpSatSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.scheduling import (
    SchedulingCpSatSolver,
)


class MultimodeSchedulingCpSatSolver(
    SchedulingCpSatSolver[Task], MultimodeCpSatSolver[Task], Generic[Task]
):
    """Base class for cpsat solvers dealing with scheduling problems whose tasks durations depend only on mode.

    Automatically managed creation of some variables
    - start
    - end
    - task-mode choice + only one to chosen constraint
    - duration
    - task intervals (constraint duration = end - start)
    - task-mode opt intervals

    """

    problem: MultimodeSchedulingProblem[Task]

    @abstractmethod
    def get_task_mode_interval(self, task: Task, mode: int) -> IntervalVar:
        """Get the interval variable corresponding to given task and mode."""
        ...
