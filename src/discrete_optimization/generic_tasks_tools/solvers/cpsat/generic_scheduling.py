#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from typing import Generic

from ortools.sat.python.cp_model import IntervalVar, LinearExprT

from discrete_optimization.generic_tasks_tools.allocation import UnaryResource
from discrete_optimization.generic_tasks_tools.base import Task
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
from discrete_optimization.generic_tasks_tools.solvers.cpsat.non_renewable_resource import (
    NonRenewableCpSatSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.precedence_scheduling import (
    PrecedenceSchedulingCpSatSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.skill import (
    SkillSchedulingCpSatSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.timelag import (
    TimelagCpSatSolver,
)


class GenericSchedulingCpSatSolver(
    SkillSchedulingCpSatSolver[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, UnaryResource
    ],
    NonRenewableCpSatSolver[Task, NonRenewableResource],
    PrecedenceSchedulingCpSatSolver[Task],
    TimelagCpSatSolver[Task],
    Generic[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, NonRenewableResource
    ],
):
    """Mixin for cpsat solver dealing with scheduling + allocation problems.

    Has access to helping methods to create constraints for
    - precedence
    - renewable resource with calendar (unary resource to allocate, or cumulative resource)
    - non-renewable resource capacity
    - skills brought to tasks by allocated unary resources

    For a more all-in-one version actually creating variables, constraints and objectives,
    see `GenericSchedulingAutoCpSatSolver`.

    """

    problem: GenericSchedulingProblem[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, NonRenewableResource
    ]

    def get_resource_consumption_intervals(
        self, resource: Resource
    ) -> list[tuple[IntervalVar, LinearExprT]]:
        if self.problem.is_unary_resource(resource=resource):
            if self.avoid_interval_optional:
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
