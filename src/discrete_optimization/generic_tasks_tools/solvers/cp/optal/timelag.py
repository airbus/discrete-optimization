#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.solvers.cp.base.timelag import (
    TimelagCpSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.optal.scheduling import (
    SchedulingOptalSolver,
)


class TimelagOptalSolver(TimelagCpSolver[Task], SchedulingOptalSolver[Task]): ...
