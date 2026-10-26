#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Generic

from discrete_optimization.generic_tasks_tools.allocation import UnaryResource
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.generic_scheduling import (
    CumulativeResource,
)
from discrete_optimization.generic_tasks_tools.solvers.base.cumulative_resource import (
    CumulativeResource,
    OtherCalendarResource,
)
from discrete_optimization.generic_tasks_tools.solvers.base.resource_usage_by_unary import (
    ResourceUsageByUnaryResourceCpBaseSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.optal.cumulative_resource import (
    CumulativeResourceSchedulingOptalSolver,
)


class ResourceUsageByUnaryResourceOptalSolver(
    ResourceUsageByUnaryResourceCpBaseSolver[
        Task, CumulativeResource, OtherCalendarResource, UnaryResource
    ],
    CumulativeResourceSchedulingOptalSolver[
        Task, CumulativeResource, OtherCalendarResource
    ],
    Generic[Task, CumulativeResource, OtherCalendarResource, UnaryResource],
): ...
