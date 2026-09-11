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
from discrete_optimization.generic_tasks_tools.solvers.cpsat.utils import (
    ModeToValueModeling,
    create_resource_dependent_variable,
    create_variable_function_of_mode_on_solver,
)


class CumulativeResourceSchedulingCpSatSolver(
    CalendarResourceCpSatSolver[Task, Resource],
    MultimodeSchedulingCpSatSolver[Task],
    Generic[Task, CumulativeResource, OtherCalendarResource],
):
    """Base class for cpsat solvers dealing with scheduling problems handling cumulative resources."""

    problem: CumulativeResourceProblem[Task, CumulativeResource, OtherCalendarResource]
    avoid_interval_optional_for_cumulative_resources: bool = False
    """Whether using task intervals + demand vars or optional intervals depending on is_present[unary_resource] in cumulative/no_overlap constraints."""
    cumulative_demand_resource_task_initialized: bool = False
    demands_resource_task: dict[tuple[CumulativeResource, Task], LinearExprT]
    demand_cumulative_resource_task_initialized: bool = False
    demands_cumulative_resource_vars: dict[tuple[CumulativeResource, Task], LinearExprT]
    demand_cumulative_modeling: ModeToValueModeling

    def get_resource_consumption_intervals(
        self, resource: Resource
    ) -> list[tuple[IntervalVar, LinearExprT]]:
        if self.problem.is_cumulative_resource(resource):
            if (
                self.avoid_interval_optional_for_cumulative_resources
                or self.problem.has_any_cumulative_consumption_dependent()
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

    def initialize_cumulative_resource_demand_vars(self):
        """
        Build either expression or variable array for resource demand.
        For task for which resource demand only depends on its own mode, this is a simple expression,
        While for dependent consumption based of other task mode, additional variable is added.
        """
        self.demands_cumulative_resource_vars = {}
        task_mode_var = {
            (t, m): self.get_task_mode_is_present_variable(task=t, mode=m)
            for t in self.problem.tasks_list
            for m in self.problem.get_task_modes(t)
        }
        for task in self.problem.tasks_list:
            for resource in self.problem.cumulative_resources_list:
                if self.problem.is_cumulative_resource_task_consumption_dependent(
                    resource=resource, task=task
                ):
                    self.demands_cumulative_resource_vars[task, resource] = (
                        create_resource_dependent_variable(
                            cp_model=self.cp_model,
                            name_var=f"conso_{task}_{resource}",
                            task=task,
                            task_mode_var=task_mode_var,
                            mode2mapping={
                                mode: self.problem.get_cumulative_resource_consumption_mapping(
                                    resource=resource, task=task, mode=mode
                                )
                                for mode in self.problem.get_task_modes(task=task)
                            },
                        )
                    )
                else:
                    mode2value = {
                        m: self.problem.get_cumulative_resource_consumption(
                            resource=resource, task=task, mode=m
                        )
                        for m in self.problem.get_task_modes(task)
                    }
                    mode2var = {
                        m: self.get_task_mode_is_present_variable(task=task, mode=m)
                        for m in self.problem.get_task_modes(task)
                    }
                    self.demands_cumulative_resource_vars[task, resource] = (
                        create_variable_function_of_mode_on_solver(
                            solver=self,
                            name=f"conso_{task}_{resource}",
                            mode2value=mode2value,
                            mode2var=mode2var,
                            modeling=self.demand_cumulative_modeling,
                        )
                    )
        self.demand_cumulative_resource_task_initialized = True

    def get_cumulative_resource_demand_variable(
        self, task: Task, resource: CumulativeResource
    ) -> LinearExprT:
        """Get the variable representing the resource demand by the task.

        Default to a linear expression using consumption per mode and is_present variables.
        If demand variables are indeed created in the cp_model, this should be overriden to return it
        so that cumulative resource constraints are constraining these variables.

        Needed if `self.avoid_interval_optional_for_cumulative_resources` is set to True.

        Args:
            task:
            resource:

        Returns:

        """
        if not self.demand_cumulative_resource_task_initialized:
            self.initialize_cumulative_resource_demand_vars()
        return self.demands_cumulative_resource_vars[task, resource]
