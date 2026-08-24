#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Hashable
from typing import Generic, Optional, TypeVar

from discrete_optimization.generic_tools.cp_tools import CpSolver
from discrete_optimization.generic_tools.do_problem import Problem, Solution

Task = TypeVar("Task", bound=Hashable)


class TasksProblem(Problem, Generic[Task]):
    """Base class for scheduling/allocation problems."""

    _map_task_to_index: Optional[dict[Task, int]] = None

    @property
    @abstractmethod
    def tasks_list(self) -> list[Task]:
        """List of all tasks to schedule or allocate to."""
        ...

    @abstractmethod
    def is_optional(self, task: Task) -> bool: ...
    @property
    def optional_tasks_list(self) -> list[Task]:
        return [t for t in self.tasks_list if self.is_optional(t)]

    def get_index_from_task(self, task: Task) -> int:
        if self._map_task_to_index is None:
            self._map_task_to_index = {
                task: i for i, task in enumerate(self.tasks_list)
            }
        return self._map_task_to_index[task]

    def get_task_from_index(self, i: int) -> Task:
        return self.tasks_list[i]

    def update_tasks_list(self) -> None:
        """To be call when tasks_list is updated to reset the cache."""
        self._map_task_to_index = None


class NoOptionalTasksProblem(TasksProblem[Task]):
    def is_optional(self, task: Task) -> bool:
        return False


class TasksSolution(Solution, Generic[Task]):
    """Base class for scheduling/allocation solutions."""

    problem: TasksProblem[Task]

    @abstractmethod
    def is_present(self, task: Task) -> bool: ...

    def check_present_tasks(self) -> bool:
        for t in self.problem.tasks_list:
            if not self.problem.is_optional(t):
                if not self.is_present(t):
                    return False
        return True

    def get_present_tasks(self):
        return [t for t in self.problem.tasks_list if self.is_present(t)]


class TasksCpSolver(CpSolver, Generic[Task]):
    """Base class for cp solver handling tasks problems."""

    problem: TasksProblem[Task]
