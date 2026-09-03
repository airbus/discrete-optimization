#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from ortools.sat.python.cp_model import LinearExpr

from discrete_optimization.generic_tasks_tools.objectives.unary_resource_used import (
    UnaryResourcesUsedComputer,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.objectives.objective_modeler import (
    ObjectiveModelerCpSat,
)


class UnaryResourcesUsedModelerCpSat(ObjectiveModelerCpSat):
    objective_computer: UnaryResourcesUsedComputer

    def get_objective_expr(self) -> LinearExpr:
        self.solver.create_used_variables()
        return sum(
            [
                self.solver.used_variables[ur] * weight
                for ur in self.solver.used_variables
                if (weight := self.objective_computer.get_weight_per_unary_resource(ur))
                > 0
            ]
        )
