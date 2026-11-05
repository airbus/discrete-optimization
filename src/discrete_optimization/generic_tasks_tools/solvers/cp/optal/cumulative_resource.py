#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Generic

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.cumulative_resource import (
    CumulativeResource,
    OtherCalendarResource,
    Resource,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.cumulative_resource import (
    CumulativeResourceSchedulingCpSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.optal.calendar_resource import (
    CalendarResourceOptalSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.optal.multimode_scheduling import (
    MultimodeSchedulingOptalSolver,
)


class CumulativeResourceSchedulingOptalSolver(
    CumulativeResourceSchedulingCpSolver[
        Task, CumulativeResource, OtherCalendarResource
    ],
    CalendarResourceOptalSolver[Task, Resource],
    MultimodeSchedulingOptalSolver[Task],
    Generic[Task, CumulativeResource, OtherCalendarResource],
):
    """Base class for solvers dealing with scheduling problems handling cumulative resources."""

    ...
