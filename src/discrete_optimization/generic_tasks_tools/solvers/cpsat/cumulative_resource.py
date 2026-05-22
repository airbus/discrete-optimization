#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Generic

from ortools.sat.python.cp_model import IntervalVar, LinearExprT

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.cumulative_resource import (
    CumulativeResource,
    CumulativeResourceProblem,
    OtherCalendarResource,
    Resource,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.calendar_resource import (
    CalendarResourceCpSatSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.multimode_scheduling import (
    MultimodeSchedulingCpSatSolver,
)


class CumulativeResourceSchedulingCpSatSolver(
    CalendarResourceCpSatSolver[Task, Resource],
    MultimodeSchedulingCpSatSolver[Task],
    Generic[Task, CumulativeResource, OtherCalendarResource],
):
    """Base class for cpsat solvers dealing with scheduling problems handling cumulative resources."""

    problem: CumulativeResourceProblem[Task, CumulativeResource, OtherCalendarResource]

    avoid_interval_optional: bool = False
    """Whether using task intervals + demand vars instead of optional intervals depending on is_present[mode]."""

    def get_resource_consumption_intervals(
        self, resource: Resource
    ) -> list[tuple[IntervalVar, LinearExprT]]:
        if self.problem.is_cumulative_resource(resource):
            if self.avoid_interval_optional:
                # no optional interval, use rather demand variables
                return [
                    (self.get_task_interval(task=task), conso)
                    for task in self.problem.tasks_list
                    if not isinstance(
                        (
                            conso := self.get_cumulative_resource_demand_variable(
                                task=task, resource=resource
                            )
                        ),
                        int,
                    )
                    or conso > 0
                ]
            else:
                return [
                    (
                        self.get_task_mode_interval(task=task, mode=mode),
                        conso,
                    )
                    for task in self.problem.tasks_list
                    for mode in self.problem.get_task_modes(task=task)
                    if (
                        conso := self.problem.get_cumulative_resource_consumption(
                            resource=resource, task=task, mode=mode
                        )
                    )
                    > 0
                ]
        else:
            raise NotImplementedError(
                f"{resource} is not a cumulative resource whose consumption depends only on task mode."
            )

    def get_cumulative_resource_demand_variable(
        self, task: Task, resource: CumulativeResource
    ) -> LinearExprT:
        """Get the variable representing the resource demand by the task.

        Default to a linear expression using consumption per mode and is_present variables.
        If demand variables are indeed created in the cp_model, this should be overriden to return it
        so that cumulative resource constraints are constraining these variables.

        Needed if `self.avoid_interval_optional` is set to True.

        Args:
            task:
            resource:

        Returns:

        """
        return sum(
            self.get_task_mode_is_present_variable(task=task, mode=mode) * conso
            for mode in self.problem.get_task_modes(task)
            if (
                conso := self.problem.get_cumulative_resource_consumption(
                    resource=resource, task=task, mode=mode
                )
            )
            > 0
        )

    def get_task_interval(self, task: Task) -> IntervalVar:
        """Return interval variable for the task.

        This variable corresponds to the task schedule, whatever the mode.
        This is needed when wanting to avoid using optional interval,
        i.e. if `self.avoid_interval_optional` is set to True.

        """
        raise NotImplementedError
