#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Generic

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.solvers.base.multimode_scheduling import (
    MultimodeSchedulingCpBaseSolver,
    SinglemodeSchedulingCpBaseSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.base.typing import (
    LinearExpr,
)
from discrete_optimization.generic_tasks_tools.solvers.optal.scheduling import (
    SchedulingOptalSolver,
)

try:
    import optalcp as cp
except ImportError:
    cp = None


class MultimodeSchedulingOptalSolver(
    MultimodeSchedulingCpBaseSolver[Task], SchedulingOptalSolver[Task], Generic[Task]
):
    """Mixin for optal solvers dealing with scheduling problems whose tasks durations depend only on mode."""

    def get_task_mode_is_present_variable(self, task: Task, mode: int) -> LinearExpr:
        itv: cp.IntervalVar = self.get_task_mode_interval(task=task, mode=mode)
        return itv.presence()


class SinglemodeSchedulingOptalSolver(
    MultimodeSchedulingOptalSolver[Task], SinglemodeSchedulingCpBaseSolver[Task]
):
    """Base class for solvers dealing with single mode scheduling problems with fixed tasks durations."""

    def get_task_mode_interval(self, task: Task, mode: int) -> cp.IntervalVar:
        return self.get_task_interval(task=task)
