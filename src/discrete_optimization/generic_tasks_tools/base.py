#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Hashable
from typing import Generic, Optional, TypeVar

from discrete_optimization.generic_tasks_tools.utils import optional_override
from discrete_optimization.generic_tools.cp_tools import CpSolver
from discrete_optimization.generic_tools.do_problem import Problem, Solution

logger = logging.getLogger(__name__)

Task = TypeVar("Task", bound=Hashable)


class TasksProblem(Problem, Generic[Task]):
    """Base class for scheduling/allocation problems."""

    _map_task_to_index: Optional[dict[Task, int]] = None

    @property
    @abstractmethod
    def tasks_list(self) -> list[Task]:
        """List of all tasks to schedule or allocate to."""
        ...

    @optional_override
    def is_optional(self, task: Task) -> bool:
        """Whether a task is optional or not.

        It means that the task can be ignored in the solution.
        If absent of the solution, it can also be removed from the constraints.

        Default to no optional task.

        """
        return False

    @property
    def optional_tasks_list(self) -> list[Task]:
        return [t for t in self.tasks_list if self.is_optional(t)]

    def has_optional_tasks(self) -> bool:
        return len(self.optional_tasks_list) > 0

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


class TasksSolution(Solution, Generic[Task]):
    """Base class for scheduling/allocation solutions."""

    problem: TasksProblem[Task]

    @optional_override
    def is_present(self, task: Task) -> bool:
        """Tell whether the task is present in the solution.

        It can mean several thing:
        - scheduling problem: start and end of the task are defined
        - allocation problem: a ressource has been allocated to the task
        - multimode: a mode has been chosen for the task
        - or a mix of it

        Sometimes a scheduling allocation problem will allow no resource allocation for present task,
        it has to be defined problem by problem.

        For convenience, a default implementation is provided which assumes that all tasks are present.
        To be overriden in subclasses.

        If and only if this method returns False, the mode, start, and end of the task can have the value
        `AbsentValue.ABSENT`.

        If the method returns False, the task is removed from all constraints during checks.

        """
        return True

    def check_present_tasks(self) -> bool:
        for task in self.problem.tasks_list:
            if not (self.problem.is_optional(task) or self.is_present(task)):
                logger.debug(
                    f"Task '{task}' is not present even though it is not optional."
                )
                return False
        return True

    def get_present_tasks(self):
        return [t for t in self.problem.tasks_list if self.is_present(t)]


class TasksCpSolver(CpSolver, Generic[Task]):
    """Base class for cp solver handling tasks problems."""

    problem: TasksProblem[Task]
