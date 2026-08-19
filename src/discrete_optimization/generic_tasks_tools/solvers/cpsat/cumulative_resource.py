#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Generic

from ortools.sat.python.cp_model import Domain, IntervalVar, LinearExprT

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
    cumulative_demand_resource_task_initialized: bool = False
    demands_resource_task: dict[tuple[CumulativeResource, Task], LinearExprT]
    avoid_interval_optional: bool = False
    """Whether using task intervals + demand vars instead of optional intervals depending on is_present[mode]."""

    def get_resource_consumption_intervals(
        self, resource: Resource
    ) -> list[tuple[IntervalVar, LinearExprT]]:
        if self.problem.is_cumulative_resource(resource):
            if (
                self.avoid_interval_optional
                or self.problem.has_any_consumption_dependent()
            ):
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

    def initialize_resource_demand_vars_and_expr(self):
        """
        Build either expression or variable array for resource demand.
        For task for which resource demand only depends on its own mode, this is a simple expression,
        While for dependent consumption based of other task mode, additional variable is added.
        """
        self.demands_resource_task = {}
        for task in self.problem.tasks_list:
            for resource in self.problem.cumulative_resources_list:
                if self.problem.is_resource_task_consumption_dependent(
                    resource=resource, task=task
                ):
                    possible_values = (
                        self.problem.get_possible_resource_consumption_all_modes(
                            task=task, resource=resource
                        )
                    )
                    self.demands_resource_task[resource, task] = (
                        self.cp_model.new_int_var_from_domain(
                            domain=Domain.FromValues(list(possible_values)),
                            name=f"conso_{task}_{resource}",
                        )
                    )
                    for mode in self.problem.get_task_modes(task=task):
                        mapping = (
                            self.problem.get_cumulative_resource_consumption_mapping(
                                resource=resource, task=task, mode=mode
                            )
                        )
                        for set_task_mode in mapping:
                            value = mapping[set_task_mode]
                            modes_var = [
                                self.get_task_mode_is_present_variable(task=tt, mode=mm)
                                for tt, mm in set_task_mode
                            ]
                            (
                                self.cp_model.add(
                                    self.demands_resource_task[resource, task] == value
                                ).only_enforce_if(
                                    *(
                                        [
                                            self.get_task_mode_is_present_variable(
                                                task=task, mode=mode
                                            )
                                        ]
                                        + modes_var
                                    )
                                )
                            )
                else:
                    self.demands_resource_task[resource, task] = sum(
                        self.get_task_mode_is_present_variable(task=task, mode=mode)
                        * conso
                        for mode in self.problem.get_task_modes(task)
                        if (
                            conso := self.problem.get_cumulative_resource_consumption(
                                resource=resource, task=task, mode=mode
                            )
                        )
                        > 0
                    )
        self.demand_resource_task_initialized = True

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
        if not self.cumulative_demand_resource_task_initialized:
            self.initialize_resource_demand_vars_and_expr()
        return self.demands_resource_task[resource, task]
