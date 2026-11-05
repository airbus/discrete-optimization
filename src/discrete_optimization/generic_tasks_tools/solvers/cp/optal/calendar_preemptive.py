#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from enum import Enum
from typing import Generic

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.calendar_preemptive import (
    CumulativeResource,
    OtherCalendarResource,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.calendar_preemptive import (
    CalendarPreemptiveCpSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.optal.multimode_scheduling import (
    MultimodeSchedulingOptalSolver,
)


class ModelingCalendarPreemptive(Enum):
    INDICATOR = 0
    ELEMENT = 1


class CalendarPreemptiveOptalSolver(
    CalendarPreemptiveCpSolver[Task, CumulativeResource, OtherCalendarResource],
    MultimodeSchedulingOptalSolver[Task],
    Generic[Task, CumulativeResource, OtherCalendarResource],
): ...
