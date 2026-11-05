#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tasks_tools.objectives.resource_levels import (
    CalendarRenewableResourceLevelObjectiveComputer,
    NonRenewableResourceLevelObjectiveComputer,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.objectives.objective_modeler import (
    ObjectiveModelerCp,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.typing import LinearExpr


class CalendarRenewableResourceLevelModelerCp(ObjectiveModelerCp):
    objective_computer: CalendarRenewableResourceLevelObjectiveComputer

    def get_objective_expr(self) -> LinearExpr:
        if not self.objective_computer.has_any_weight():
            return 0
        else:
            self.solver._create_resource_level_variables()
            return self.solver.sum(
                self.solver.resource_level_variables[res]
                * self.objective_computer.get_weight_resource(res)
                for res in self.objective_computer.weight_resource
                if self.objective_computer.get_weight_resource(res) > 0
            )


class NonRenewableResourceLevelModelerCp(ObjectiveModelerCp):
    objective_computer: NonRenewableResourceLevelObjectiveComputer

    def get_objective_expr(self) -> LinearExpr:
        if not self.objective_computer.has_any_weight():
            return 0
        else:
            self.solver._create_resource_level_variables()
            return self.solver.sum(
                self.solver.resource_level_variables[res]
                * self.objective_computer.get_weight_resource(res)
                for res in self.objective_computer.weight_resource
                if self.objective_computer.get_weight_resource(res) > 0
            )
