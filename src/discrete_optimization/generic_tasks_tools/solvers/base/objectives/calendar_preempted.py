#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import TYPE_CHECKING

from discrete_optimization.generic_tasks_tools.solvers.base.typing import LinearExpr

if TYPE_CHECKING:
    from discrete_optimization.generic_tasks_tools.solvers.base.auto import (
        GenericSchedulingAutoCpBaseSolver,
    )
from discrete_optimization.generic_tasks_tools.objectives.calendar_preempted import (
    CalendarPreemptedComputer,
)
from discrete_optimization.generic_tasks_tools.solvers.base.objectives.objective_modeler import (
    ObjectiveModelerCpBase,
)


class CalendarPreemptedModelerCpBase(ObjectiveModelerCpBase):
    objective_computer: CalendarPreemptedComputer
    solver: GenericSchedulingAutoCpBaseSolver

    def get_objective_expr(self) -> LinearExpr:
        return self.solver.compute_nb_preempted_tasks()
