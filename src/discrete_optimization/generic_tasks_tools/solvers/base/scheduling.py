#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Iterable

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.scheduling import SchedulingProblem
from discrete_optimization.generic_tasks_tools.solvers.base.base import TasksSolver
from discrete_optimization.generic_tools.enums import SignEnum


class SchedulingSolver(TasksSolver[Task]):
    """Base class for cp solvers handling scheduling problems."""

    problem: SchedulingProblem[Task]

    def get_makespan_lower_bound(self) -> int:
        """Get a lower bound on global makespan.

        Can be overriden in solvers wanting to specify it in init_model() for instance.

        """
        return self.problem.get_makespan_lower_bound()

    def get_makespan_upper_bound(self) -> int:
        """Get a upper bound on global makespan."""
        return self.problem.get_makespan_upper_bound()

    @abstractmethod
    def add_constraint_on_task(
        self, task: Task, start_or_end: StartOrEnd, sign: SignEnum, time: int
    ) -> list[Any]:
        """Add constraint on given task start or end

        task start or end must compare to `time` according to `sign`

        Args:
            task:
            start_or_end:
            sign:
            time:

        Returns:
            resulting constraints
        """
        ...

    @abstractmethod
    def add_constraint_chaining_tasks(self, task1: Task, task2: Task) -> list[Any]:
        """Add constraint chaining task1 with task2

        task2 start == task1 end

        Args:
            task1:
            task2:

        Returns:
            resulting constraints

        """
        ...

    def get_global_makespan_variable(self) -> Any:
        """Construct and get the variable tracking the global makespan.

        Default implementation uses `get_subtasks_makespan_variable` on last tasks.
        Beware: a further call to `get_subtasks_makespan_variable` with another subset of tasks can
        change the constraints on this variable and thus make it obsolete.

        Returns:
            objective variable to minimize

        """
        return self.get_subtasks_makespan_variable(
            subtasks=set(self.problem.get_last_tasks())
        )

    @abstractmethod
    def get_subtasks_makespan_variable(self, subtasks: Iterable[Task]) -> Any:
        """Construct and get the variable tracking the makespan on a subset of tasks.

        Beware: a further call to `get_subtasks_makespan_variable` with another subset of tasks can
        change the constraints on this variable and thus make it obsolete.

        Args:
            subtasks:

        Returns:
            objective variable to minimize

        """
        ...

    @abstractmethod
    def get_subtasks_sum_end_time_variable(self, subtasks: Iterable[Task]) -> Any:
        """Construct and get the variable tracking the sum of end times on a subset of tasks.

        Args:
            subtasks:

        Returns:
            objective variable to minimize

        """
        ...

    @abstractmethod
    def get_subtasks_sum_start_time_variable(self, subtasks: Iterable[Task]) -> Any:
        """Construct and get the variable tracking the sum of start times on a subset of tasks.

        Args:
            subtasks:

        Returns:
            objective variable to minimize

        """
        ...
