#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Generic

from discrete_optimization.generic_tasks_tools.allocation import UnaryResource
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.cumulative_resource import (
    OtherCalendarResource,
)
from discrete_optimization.generic_tasks_tools.skill import (
    CumulativeResource,
    NonSkillCumulativeResource,
    Skill,
)
from discrete_optimization.generic_tasks_tools.solvers.base.skill import (
    SkillSchedulingCpBaseSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.optal.calendar_resource_generic import (
    CalendarResourceGenericOptalSolver,
)


class SkillSchedulingOptalSolver(
    SkillSchedulingCpBaseSolver[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, OtherCalendarResource
    ],
    CalendarResourceGenericOptalSolver[
        Task, CumulativeResource, OtherCalendarResource, UnaryResource
    ],
    Generic[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, OtherCalendarResource
    ],
): ...
