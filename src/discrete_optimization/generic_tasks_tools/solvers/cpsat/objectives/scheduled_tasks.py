#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from ortools.sat.python.cp_model import LinearExpr

from discrete_optimization.generic_tasks_tools.objectives.scheduled_tasks import (
    ScheduledTasksComputer,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.objectives.objective_modeler import (
    ObjectiveModelerCpSat,
)


class ScheduledTasksCpSatModeler(ObjectiveModelerCpSat):
    objective_computer: ScheduledTasksComputer

    def get_objective_expr(self) -> LinearExpr:
        return sum(
            self.objective_computer.weight_per_task[t]
            * self.solver.get_task_is_present_variable(t)
            for t in self.objective_computer.weight_per_task
        )
