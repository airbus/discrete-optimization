#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from typing import Generic

from discrete_optimization.generic_tasks_tools.allocation import UnaryResource
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.cumulative_resource import (
    OtherCalendarResource,
)
from discrete_optimization.generic_tasks_tools.exclusions import ExclusionResource
from discrete_optimization.generic_tasks_tools.generic_scheduling import (
    GenericSchedulingProblem,
    Resource,
)
from discrete_optimization.generic_tasks_tools.non_renewable_resource import (
    NonRenewableResource,
)
from discrete_optimization.generic_tasks_tools.skill import (
    NonSkillCumulativeResource,
    Skill,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.alternative_subproblems import (
    AlternativeSubproblemCpSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.exclusions import (
    ExclusionCpSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.no_overlap import (
    NoOverlapCpSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.non_renewable_resource import (
    NonRenewableCpSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.precedence_scheduling import (
    PrecedenceSchedulingCpSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.skill import (
    SkillSchedulingCpSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.timelag import (
    TimelagCpSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cp.base.typing import (
    IntervalVar,
    LinearExpr,
)


class GenericSchedulingCpSolver(
    ExclusionCpSolver[Task, ExclusionResource],
    AlternativeSubproblemCpSolver[Task],
    NoOverlapCpSolver[Task],
    TimelagCpSolver[Task],
    PrecedenceSchedulingCpSolver[Task],
    NonRenewableCpSolver[Task, NonRenewableResource, UnaryResource],
    SkillSchedulingCpSolver[
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
    """Mixin for cp solver dealing with scheduling + allocation problems.

    Has access to helping methods to create constraints for
    - precedence
    - renewable resource with calendar (unary resource to allocate, or cumulative resource)
    - non-renewable resource capacity
    - skills brought to tasks by allocated unary resources
    - resource blocking (gaps and spans)

    For a more all-in-one version actually creating variables, constraints and objectives,
    see `GenericSchedulingAutoCpSolver`.

    """

    problem: GenericSchedulingProblem[
        Task,
        UnaryResource,
        Skill,
        NonSkillCumulativeResource,
        NonRenewableResource,
        ExclusionResource,
    ]

    avoid_interval_optional_for_unary_resources: bool = False
    """Whether using task intervals + is_present(unary_resource) as demand variables or optional intervals depending on is_present(unary_resource) in cumulative/no_overlap constraints."""

    def get_resource_consumption_intervals(
        self, resource: Resource
    ) -> list[tuple[IntervalVar, LinearExpr]]:
        if self.problem.is_unary_resource(resource=resource):
            if self.avoid_interval_optional_for_unary_resources:
                # no optional interval, use rather demand variables
                return [
                    (self.get_task_interval(task=task), conso)
                    for task in self.problem.tasks_list
                    if not isinstance(
                        (
                            conso := self.get_task_unary_resource_is_present_variable(
                                task=task, unary_resource=resource
                            )
                        ),
                        int,
                    )
                    or conso > 0
                ]
            else:
                return [
                    (
                        self.get_task_unary_resource_interval(
                            task=task, unary_resource=resource
                        ),
                        1,
                    )
                    for task in self.problem.tasks_list
                    if self.is_compatible_task_unary_resource(
                        task=task, unary_resource=resource
                    )
                ]
        else:
            # For cumulative resources, delegate to parent
            # ResourceBlockingCpSolver will add blocking intervals
            return super().get_resource_consumption_intervals(resource=resource)

    @abstractmethod
    def get_task_unary_resource_interval(
        self, task: Task, unary_resource: UnaryResource
    ) -> IntervalVar:
        """Get the interval variable corresponding to given task conditioned to allocation of the given unary resource.

        The method may return an error (no variable existing) if
        `self.problem.is_compatible_task_unary_resource(task=task, unary_resource=unary_resource)` is false.

        """
        ...

    def get_makespan_upper_bound(self) -> int:
        return self.problem.get_makespan_tighter_upper_bound()
