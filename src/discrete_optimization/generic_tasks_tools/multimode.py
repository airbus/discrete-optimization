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
    def is_multimode(self) -> bool:
        return self.max_number_of_mode > 1

    @property
    def max_number_of_mode(self) -> int:
        return max(len(self.get_task_modes(task)) for task in self.tasks_list)


class SinglemodeProblem(MultimodeProblem[Task]):
    @property
    def default_mode(self):
        """Default single mode.

        To be overriden when default value has more sense with another value (ex: in rcpsp, default mode is 1)

        """
        return 0

    def get_task_modes(self, task: Task) -> set[int]:
        return {self.default_mode}

    @property
    def is_multimode(self) -> bool:
        return True

    @property
    def max_number_of_mode(self) -> int:
        return 1


class SinglemodeSolution(MultimodeSolution[Task]):
    problem: SinglemodeProblem[Task]

    def get_mode(self, task: Task) -> int:
        return self.problem.default_mode


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
