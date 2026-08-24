#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from ortools.sat.python.cp_model import LinearExpr

from discrete_optimization.generic_tasks_tools.objectives.mode_cost import (
    ModeCostComputer,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.objectives.objective_modeler import (
    ObjectiveModelerCpSat,
)


class ModeCostModelerCpSat(ObjectiveModelerCpSat):
    objective_computer: ModeCostComputer

    def get_objective_expr(self) -> LinearExpr:
        if not self.objective_computer.has_any_mode_cost():
            return 0
        return sum(
            self.solver.get_task_mode_is_present_variable(task, mode) * mode_cost
            for task in self.solver.problem.tasks_list
            for mode in self.solver.problem.get_task_modes(task)
            if (mode_cost := self.objective_computer.mode_cost(task, mode)) != 0
        )
