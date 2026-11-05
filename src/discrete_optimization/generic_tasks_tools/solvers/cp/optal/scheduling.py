#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import abstractmethod
from typing import Iterable

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.solvers.cp.base.scheduling import (
    SchedulingCpSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.typing import (
    LinearExpr,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.optal.base import (
    TasksOptalSolver,
)

try:
    import optalcp as cp
except ImportError:
    cp = None


class SchedulingOptalSolver(TasksOptalSolver[Task], SchedulingCpSolver[Task]):
    """Mixin for optal solvers handling scheduling problems."""

    def get_task_start_or_end_variable(
        self, task: Task, start_or_end: StartOrEnd
    ) -> LinearExpr:
        """Retrieve the variable storing the start or end time of given task.

        For optional tasks, the start and end should be 0/False when absent
        so that the makespan/sum of starts/sum of ends variables do not take them into account.

        Args:
            task:
            start_or_end:

        Returns:

        """
        itv = self.get_task_interval(task)
        if start_or_end == StartOrEnd.START:
            return itv.start()
        else:
            return itv.end()

    @abstractmethod
    def get_task_interval(self, task: Task) -> cp.IntervalVar:
        """Return interval variable for the task.

        This variable corresponds to the task schedule, whatever the mode.
        This is the main optal variable (start, end and duration are derived from it)

        If the task is optional, this interval should be also taken into account
        the presence/absence of the task.

        """
        pass

    def get_duration_variable(self, task: Task) -> LinearExpr:
        return self.get_task_interval(task).length()

    def get_global_makespan_variable(self) -> LinearExpr:
        return self.cp_model.max(
            tuple(
                self.get_task_start_or_end_variable(task, StartOrEnd.END)
                for task in self.problem.get_last_tasks()
            )
        )

    def get_subtasks_makespan_variable(self, subtasks: Iterable[Task]) -> LinearExpr:
        return self.cp_model.max(
            tuple(
                self.get_task_start_or_end_variable(task, StartOrEnd.END)
                for task in subtasks
            )
        )

    def get_subtasks_sum_end_time_variable(
        self, subtasks: Iterable[Task]
    ) -> LinearExpr:
        return self.cp_model.sum(
            tuple(
                self.get_task_start_or_end_variable(task, StartOrEnd.END)
                for task in subtasks
            )
        )

    def get_subtasks_sum_start_time_variable(
        self, subtasks: Iterable[Task]
    ) -> LinearExpr:
        return self.cp_model.sum(
            tuple(
                self.get_task_start_or_end_variable(task, StartOrEnd.START)
                for task in subtasks
            )
        )
