#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Generic

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.solvers.cp.base.alternative_subproblems import (
    AlternativeSubproblemCpSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.optal.multimode_scheduling import (
    MultimodeSchedulingOptalSolver,
)


class AlternativeSubproblemOptalSolver(
    AlternativeSubproblemCpSolver[Task],
    MultimodeSchedulingOptalSolver[Task],
    Generic[Task],
): ...
