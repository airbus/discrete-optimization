#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Generic

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.solvers.base.alternative_subproblems import (
    AlternativeSubproblemCpBaseSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.optal.multimode_scheduling import (
    MultimodeSchedulingOptalSolver,
)


class AlternativeSubproblemOptalSolver(
    AlternativeSubproblemCpBaseSolver[Task],
    MultimodeSchedulingOptalSolver[Task],
    Generic[Task],
): ...
