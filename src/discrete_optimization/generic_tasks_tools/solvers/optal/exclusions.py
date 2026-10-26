#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from typing import Generic

from discrete_optimization.generic_tasks_tools.exclusions import (
    ExclusionResource,
    Task,
)
from discrete_optimization.generic_tasks_tools.solvers.base.exclusions import (
    ExclusionCpBaseSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.optal.multimode_scheduling import (
    MultimodeSchedulingOptalSolver,
)

logger = logging.getLogger(__name__)


class ExclusionOptalSolver(
    ExclusionCpBaseSolver[Task, ExclusionResource],
    MultimodeSchedulingOptalSolver[Task],
    Generic[Task, ExclusionResource],
): ...
