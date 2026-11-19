#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.generic_tasks_tools.objectives.custom import CustomComputer
from discrete_optimization.generic_tasks_tools.solvers.cp.base.objectives.objective_modeler import (
    ObjectiveModelerCp,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.typing import LinearExpr


class CustomModelerCp(ObjectiveModelerCp):
    objective_computer: CustomComputer

    def get_objective_expr(self) -> LinearExpr:
        return self.solver.get_custom_objective_expr()
