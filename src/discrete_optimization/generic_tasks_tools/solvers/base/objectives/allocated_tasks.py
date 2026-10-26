#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tasks_tools.solvers.base.objectives.objective_modeler import (
    ObjectiveModelerCpBase,
)
from discrete_optimization.generic_tasks_tools.solvers.base.typing import LinearExpr


class AllocatedTasksObjectiveCpBaseModeler(ObjectiveModelerCpBase):
    def get_objective_expr(self) -> LinearExpr:
        self.solver.create_task_allocated_variables()
        return self.solver.get_nb_tasks_allocated_variable()
