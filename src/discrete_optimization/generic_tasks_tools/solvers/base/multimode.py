#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.multimode import MultimodeProblem
from discrete_optimization.generic_tasks_tools.solvers.base.base import TasksSolver


class MultimodeSolver(TasksSolver[Task]):
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
