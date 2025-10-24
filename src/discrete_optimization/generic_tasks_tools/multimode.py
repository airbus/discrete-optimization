from __future__ import annotations

from abc import abstractmethod
from typing import Any

from discrete_optimization.generic_tasks_tools.base import (
    Task,
    TasksCpSolver,
    TasksProblem,
    TasksSolution,
)


class MultimodeSolution(TasksSolution[Task]):
    """Class inherited by a solution exposing tasks modes."""

    problem: MultimodeProblem[Task]

    @abstractmethod
    def get_mode(self, task: Task) -> int:
        """Retrieve mode found for given task.

        Args:
            task:

        Returns:

        """
        ...


class MultimodeProblem(TasksProblem[Task]):
    """Class inherited by a solution exposing tasks modes."""

    @abstractmethod
    def get_task_modes(self, task: Task) -> set[int]:
        """Retrieve mode found for given task.

        Args:
            task:

        Returns:

        """
        ...

    @property
    def is_multimode(self):
        return self.max_number_of_mode > 1

    @property
    def max_number_of_mode(self):
        return max(len(self.get_task_modes(task)) for task in self.tasks_list)


class MultimodeCpSolver(TasksCpSolver[Task]):
    """Class inherited by a solver managing constraints on tasks modes."""

    problem: MultimodeProblem[Task]

    @abstractmethod
    def add_constraint_on_task_mode(self, task: Task, mode: int) -> list[Any]:
        """Add constraint on task mode

        The mode of `task` is fixed to `mode`.

        Args:
            task:
            mode:

        Returns:
            resulting constraints

        """
        ...
