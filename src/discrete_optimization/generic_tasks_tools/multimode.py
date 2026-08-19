from __future__ import annotations

from abc import abstractmethod
from enum import Enum
from typing import Any

from discrete_optimization.generic_tasks_tools.base import (
    Task,
    TasksCpSolver,
    TasksProblem,
    TasksSolution,
)


class ModeConstraintType(Enum):
    SORTED_IMPLICATION = 0
    UNORDERED = 1


import logging

logger = logging.getLogger(__name__)


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

    def check_mode_constraint(self) -> bool:
        if len(self.problem.get_mode_constraints()) == 0:
            return True
        for constraint in self.problem.get_mode_constraints():
            # First task that
            mode_constraint = constraint[0]
            list_task_mode = constraint[1]
            if mode_constraint == ModeConstraintType.SORTED_IMPLICATION:
                t0, m0 = list_task_mode[0]
                if self.get_mode(t0) == m0:
                    for i in range(1, len(list_task_mode)):
                        t, m = list_task_mode[i]
                        if self.get_mode(t) != m:
                            logger.debug(
                                f"Mode constraint not satisfied, mode of {t}!={m}"
                            )
                            return False
            if mode_constraint == ModeConstraintType.UNORDERED:
                b = (not any(self.get_mode(t) == m for t, m in list_task_mode)) or all(
                    self.get_mode(t) == m for t, m in list_task_mode
                )
                if not b:
                    logger.debug(f"Mode constraint not satisfied, {list_task_mode}")
        return True


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

    def get_mode_constraints(
        self,
    ) -> list[tuple[ModeConstraintType, list[tuple[Task, int]]]]:
        """
        An element of the list is a tuple of (ModeConstraintType, list of (task,mode))
        that implies the other choice of mode.
        For example (SORTED_IMPLICATION, [(T1, 1), (T2, 2), (T3, 1)]) means :
        if T1 is in mode 1, T2 is in mode 2, T3 is in mode 1..
        This can be useful to model mode choice that has an influence on the
        future mode choice. For example, in an assembly line
        if we choose a given station path for a product, it should stay on it !
        if mode_constraint_type == ModeConstraintType.SORTED_IMPLICATION:
            then the constraint is not active only when T1 is in mode 1,
            it should be true if any of the task,mode is active.
            So if mode(T2)==2 then the other mode are also forced!
        :return:
        """
        return []


class WithoutModeConstraintMultimodeProblem(MultimodeProblem[Task]):
    def get_mode_constraints(
        self,
    ) -> list[tuple[ModeConstraintType, list[tuple[Task, int]]]]:
        return []


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
        return False

    @property
    def max_number_of_mode(self) -> int:
        return 1


class WithoutModeConstraintSingleModeProblem(
    SinglemodeProblem[Task], WithoutModeConstraintMultimodeProblem[Task]
):
    pass


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
