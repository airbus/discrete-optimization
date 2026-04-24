#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Iterable

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.scheduling import SchedulingCpSolver
from discrete_optimization.generic_tools.cp_tools import SignEnum
from discrete_optimization.generic_tools.hub_solver.optal.optalcp_tools import (
    OptalCpSolver,
)

try:
    import optalcp as cp
except ImportError:
    cp = None
    optalcp_available = False
else:
    optalcp_available = True


class SchedulingOptalSolver(OptalCpSolver, SchedulingCpSolver[Task]):
    @abstractmethod
    def get_task_interval_variable(self, task: Task) -> "cp.IntervalVar":
        """Retrieve the interval variable of given task."""
        ...

    def get_task_start_or_end_variable(
        self, task: Task, start_or_end: StartOrEnd
    ) -> "cp.IntExpr":
        """Retrieve the variable storing the start or end time of given task.

        Args:
            task:
            start_or_end:

        Returns:

        """
        itv = self.get_task_interval_variable(task)
        if start_or_end == StartOrEnd.START:
            return self.cp_model.start(itv)
        if start_or_end == StartOrEnd.END:
            return self.cp_model.end(itv)
        return None

    def add_constraint_on_task(
        self, task: Task, start_or_end: StartOrEnd, sign: SignEnum, time: int
    ) -> list["cp.BoolExpr"]:
        var = self.get_task_start_or_end_variable(task, start_or_end)
        return self.add_bound_constraint(var, sign, time)

    def add_constraint_chaining_tasks(self, task1: Task, task2: Task) -> list[Any]:
        itv1 = self.get_task_interval_variable(task1)
        itv2 = self.get_task_interval_variable(task2)
        return [self.cp_model.start_at_end(itv2, itv1)]

    def get_subtasks_makespan_variable(self, subtasks: Iterable[Task]) -> Any:
        return self.cp_model.max(
            [
                self.get_task_start_or_end_variable(
                    task=task, start_or_end=StartOrEnd.END
                )
                for task in subtasks
            ]
        )

    def get_subtasks_sum_end_time_variable(self, subtasks: Iterable[Task]) -> Any:
        return self.cp_model.sum(
            [
                self.get_task_start_or_end_variable(
                    task=task, start_or_end=StartOrEnd.END
                )
                for task in subtasks
            ]
        )

    def get_subtasks_sum_start_time_variable(self, subtasks: Iterable[Task]) -> Any:
        return self.cp_model.max(
            [
                self.get_task_start_or_end_variable(
                    task=task, start_or_end=StartOrEnd.START
                )
                for task in subtasks
            ]
        )
