#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from ortools.sat.python.cp_model import LinearExpr

from discrete_optimization.generic_tasks_tools.solvers.cpsat.objectives.objective_modeler import (
    ObjectiveModelerCpSat,
)


class MakespanObjectiveModelCpSat(ObjectiveModelerCpSat):
    def get_objective_expr(self) -> LinearExpr:
        return self.solver.get_global_makespan_variable()
