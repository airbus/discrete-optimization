#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  This is where the real cumulative constraint is happening
#  It is there to handle the resource blocking,
#  the calendar preemptive,
#  the classic ones.
from typing import Generic

from discrete_optimization.generic_tasks_tools.allocation import UnaryResource
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.cumulative_resource import (
    OtherCalendarResource,
)
from discrete_optimization.generic_tasks_tools.skill import NonSkillCumulativeResource
from discrete_optimization.generic_tasks_tools.solvers.cp.base.calendar_resource_generic import (
    CalendarResourceGenericCpSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.optal.calendar_preemptive import (
    CalendarPreemptiveOptalSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.optal.resource_blocking import (
    ResourceBlockingOptalSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.optal.resource_usage_by_unary import (
    ResourceUsageByUnaryResourceOptalSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.optal.scheduling import (
    SchedulingOptalSolver,
)


class CalendarResourceGenericOptalSolver(
    CalendarResourceGenericCpSolver[
        Task, NonSkillCumulativeResource, OtherCalendarResource, UnaryResource
    ],
    CalendarPreemptiveOptalSolver[
        Task, NonSkillCumulativeResource, OtherCalendarResource
    ],
    ResourceBlockingOptalSolver[Task, NonSkillCumulativeResource, UnaryResource],
    ResourceUsageByUnaryResourceOptalSolver[
        Task, NonSkillCumulativeResource, OtherCalendarResource, UnaryResource
    ],
    SchedulingOptalSolver[Task],
    Generic[Task, NonSkillCumulativeResource, OtherCalendarResource, UnaryResource],
): ...
