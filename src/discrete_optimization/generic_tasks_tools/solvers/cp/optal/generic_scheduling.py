#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Generic

from discrete_optimization.generic_tasks_tools.allocation import UnaryResource
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.cumulative_resource import (
    OtherCalendarResource,
)
from discrete_optimization.generic_tasks_tools.exclusions import ExclusionResource
from discrete_optimization.generic_tasks_tools.non_renewable_resource import (
    NonRenewableResource,
)
from discrete_optimization.generic_tasks_tools.skill import (
    NonSkillCumulativeResource,
    Skill,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.generic_scheduling import (
    GenericSchedulingCpSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.typing import (
    LinearExpr,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.optal.alternative_subproblems import (
    AlternativeSubproblemOptalSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.optal.exclusions import (
    ExclusionOptalSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.optal.no_overlap import (
    NoOverlapOptalSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.optal.precedence_scheduling import (
    PrecedenceSchedulingOptalSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.optal.routing import (
    RoutingOptalSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.optal.skill import (
    SkillSchedulingOptalSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.optal.timelag import (
    TimelagOptalSolver,
)

try:
    import optalcp as cp
except ImportError:
    cp = None


class GenericSchedulingOptalSolver(
    GenericSchedulingCpSolver[
        Task,
        UnaryResource,
        Skill,
        NonSkillCumulativeResource,
        NonRenewableResource,
        OtherCalendarResource,
        ExclusionResource,
    ],
    RoutingOptalSolver[Task],
    ExclusionOptalSolver[Task, ExclusionResource],
    AlternativeSubproblemOptalSolver[Task],
    NoOverlapOptalSolver[Task],
    TimelagOptalSolver[Task],
    PrecedenceSchedulingOptalSolver[Task],
    SkillSchedulingOptalSolver[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, OtherCalendarResource
    ],
    Generic[
        Task,
        UnaryResource,
        Skill,
        NonSkillCumulativeResource,
        NonRenewableResource,
        OtherCalendarResource,
        ExclusionResource,
    ],
):
    """Mixin for optal solver dealing with scheduling + allocation problems.

    Has access to helping methods to create constraints for
    - precedence
    - renewable resource with calendar (unary resource to allocate, or cumulative resource)
    - non-renewable resource capacity
    - skills brought to tasks by allocated unary resources
    - resource blocking (gaps and spans)
    - entity constraints (precedence, time lags, time windows on any SchedulingEntity)
    - batching (variable task-to-batch assignment)

    For a more all-in-one version actually creating variables, constraints and objectives,
    see `GenericSchedulingAutoOptalSolver`.

    """

    def get_task_unary_resource_is_present_variable(
        self, task: Task, unary_resource: UnaryResource
    ) -> LinearExpr:
        itv: cp.IntervalVar = self.get_task_unary_resource_interval(
            task=task, unary_resource=unary_resource
        )
        return itv.presence()
