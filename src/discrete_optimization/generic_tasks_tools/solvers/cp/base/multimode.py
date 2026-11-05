#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from typing import Any

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.multimode import (
    ModeConstraintType,
    SinglemodeProblem,
)
from discrete_optimization.generic_tasks_tools.solvers.base.multimode import (
    MultimodeSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.base import (
    TasksCpSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.typing import LinearExpr


class MultimodeCpSolver(TasksCpSolver[Task], MultimodeSolver[Task]):
    @abstractmethod
    def get_task_mode_is_present_variable(self, task: Task, mode: int) -> LinearExpr:
        """Retrieve the boolean variable/expression telling if the mode is used for the task.

        The variable value should be 0 / False when the task is optional and absent.


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
                constraints.append(self.add_bool_constraint(var == True))
            else:
                constraints.append(self.add_bool_constraint(var == False))
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
                self.add_implication_bool_and(vars[0], vars)
                self.add_implication(vars[0], self.sum(vars) == len(vars))
                for i in range(1, len(vars)):
                    self.add_implication(vars[i - 1], vars[i])
            if mode_constraint == ModeConstraintType.UNORDERED:
                or_ = self.create_bool_var(f"constraint_mode_active_{i}")
                self.add_implication_bool_and(or_, vars)
                self.add_implication(or_, self.sum(vars) == len(vars))
                self.add_implication(~or_, self.sum(vars) == 0)
        self.add_mode_subset_constraints()

    def add_mode_subset_constraints(self):
        for i, constraint in enumerate(self.problem.get_mode_subset_constraints()):
            mode_constraint, list_task_mode = constraint
            vars = [self.create_bool_var(f"{i}_task_{t}_") for t, _ in list_task_mode]
            for k in range(len(vars)):
                task = list_task_mode[k][0]
                if self.problem.is_optional(task):
                    # Deactivate the constraint when not present
                    self.add_bool_constraint(
                        vars[k]
                        == self.sum(
                            [
                                self.get_task_mode_is_present_variable(
                                    task=list_task_mode[k][0], mode=m
                                )
                                for m in list_task_mode[k][1]
                            ]
                        )
                        + ~self.get_task_is_present_variable(task)
                    )
                else:
                    self.add_bool_constraint(
                        vars[k]
                        == self.sum(
                            [
                                self.get_task_mode_is_present_variable(
                                    task=list_task_mode[k][0], mode=m
                                )
                                for m in list_task_mode[k][1]
                            ]
                        )
                    )
            if mode_constraint == ModeConstraintType.SORTED_IMPLICATION:
                # All true if vars[0] is true.
                self.add_implication_bool_and(vars[0], vars)
                self.add_implication(vars[0], self.sum(vars) == len(vars))
                for k in range(1, len(vars)):
                    self.add_implication(vars[k - 1], vars[k])
            if mode_constraint == ModeConstraintType.UNORDERED:
                or_ = self.create_bool_var(f"constraint_mode_active_{i}")
                self.add_implication_bool_and(or_, vars)
                self.add_implication(or_, self.sum(vars) == len(vars))
                self.add_implication(~or_, self.sum(vars) == 0)


class SinglemodeCpSolver(MultimodeCpSolver[Task]):
    """Solver mixin for single mode problems."""

    problem: SinglemodeProblem[Task]

    def get_task_mode_is_present_variable(self, task: Task, mode: int) -> LinearExpr:
        return 1
