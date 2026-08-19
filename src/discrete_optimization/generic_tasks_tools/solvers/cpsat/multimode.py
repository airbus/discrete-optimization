#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from typing import Any

from ortools.sat.python.cp_model import LinearExprT

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.multimode import (
    ModeConstraintType,
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

    def add_mode_constraints(self):
        for i, constraint in enumerate(self.problem.get_mode_constraints()):
            mode_constraint, list_task_mode = constraint
            vars = [
                self.get_task_mode_is_present_variable(task=t, mode=m)
                for t, m in list_task_mode
            ]
            if mode_constraint == ModeConstraintType.SORTED_IMPLICATION:
                # All true if vars[0] is true.
                self.cp_model.AddBoolAnd(vars).only_enforce_if(vars[0])
                self.cp_model.add(sum(vars) == len(vars)).only_enforce_if(vars[0])
                for i in range(1, len(vars)):
                    self.cp_model.add_implication(vars[i - 1], vars[i])
            if mode_constraint == ModeConstraintType.UNORDERED:
                or_ = self.cp_model.NewBoolVar(f"constraint_mode_active_{i}")
                self.cp_model.add(sum(vars) == len(vars)).only_enforce_if(or_)
                self.cp_model.add(sum(vars) == 0).only_enforce_if(or_.Not())


class SinglemodeCpSatSolver(MultimodeCpSatSolver[Task]):
    """Cpsat solver mixin for single mode problems."""

    problem: SinglemodeProblem[Task]

    def get_task_mode_is_present_variable(self, task: Task, mode: int) -> LinearExprT:
        return 1
