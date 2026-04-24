#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from typing import Any

from ortools.sat.python.cp_model import LinearExprT

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.multimode import (
    MultimodeCpSolver,
    SinglemodeProblem,
)
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver


class MultimodeCpSatSolver(OrtoolsCpSatSolver, MultimodeCpSolver[Task]):
    @abstractmethod
    def get_task_mode_is_present_variable(self, task: Task, mode: int) -> LinearExprT:
        """Retrieve the 0-1 variable/expression telling if the mode is used for the task.

        Args:
            task:
            mode:

        Returns:

        """
        ...

    def add_constraint_on_task_mode(self, task: Task, mode: int) -> list[Any]:
        possible_modes = self.problem.get_task_modes(task)
        if mode not in possible_modes:
            raise ValueError(f"Task {task} cannot be done with mode {mode}.")
        if len(possible_modes) == 1:
            return []
        constraints = []
        for other_mode in possible_modes:
            var = self.get_task_mode_is_present_variable(task=task, mode=other_mode)
            if other_mode == mode:
                constraints.append(self.cp_model.add(var == True))
            else:
                constraints.append(self.cp_model.add(var == False))
        return constraints


class SinglemodeCpSatSolver(MultimodeCpSatSolver[Task]):
    """Cpsat solver mixin for single mode problems."""

    problem: SinglemodeProblem[Task]

    def get_task_mode_is_present_variable(self, task: Task, mode: int) -> LinearExprT:
        return 1
