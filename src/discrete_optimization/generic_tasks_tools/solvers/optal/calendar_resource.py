#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Generic

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.calendar_resource import (
    Resource,
)
from discrete_optimization.generic_tasks_tools.solvers.base.calendar_resource import (
    CalendarResourceCpBaseSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.optal.scheduling import (
    SchedulingOptalSolver,
)


class CalendarResourceOptalSolver(
    CalendarResourceCpBaseSolver[Task, Resource],
    SchedulingOptalSolver[Task],
    Generic[Task, Resource],
): ...
