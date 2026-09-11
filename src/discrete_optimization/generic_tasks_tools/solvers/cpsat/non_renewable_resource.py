#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Generic

from ortools.linear_solver.python.model_builder import LinearExprT

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.non_renewable_resource import (
    NonRenewableResource,
    NonRenewableResourceProblem,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.multimode import (
    MultimodeCpSatSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.utils import (
    ModeToValueModeling,
    create_resource_dependent_variable,
    create_variable_function_of_mode_on_solver,
)


class NonRenewableCpSatSolver(
    MultimodeCpSatSolver[Task], Generic[Task, NonRenewableResource]
):
    """Base class for cpsat solvers dealing with problem with non-renewable resources."""

    problem: NonRenewableResourceProblem
    demands_non_renewable_resource_initialized: bool = False
    demands_non_renewable_resource_vars: dict[
        tuple[Task, NonRenewableResource], LinearExprT
    ]
    demand_non_renewable_modeling: ModeToValueModeling

    def initialize_non_renewable_resource_demand_vars(self):
        """
        Build either expression or variable array for resource demand.
        For task for which resource demand only depends on its own mode, this is a simple expression,
        While for dependent consumption based of other task mode, additional variable is added.
        """
        self.demands_non_renewable_resource_vars = {}
        task_mode_var = {
            (t, m): self.get_task_mode_is_present_variable(task=t, mode=m)
            for t in self.problem.tasks_list
            for m in self.problem.get_task_modes(t)
        }
        for task in self.problem.tasks_list:
            for resource in self.problem.non_renewable_resources_list:
                if self.problem.is_non_renewable_resource_task_consumption_dependent(
                    resource=resource, task=task
                ):
                    self.demands_non_renewable_resource_vars[task, resource] = (
                        create_resource_dependent_variable(
                            cp_model=self.cp_model,
                            name_var=f"conso_{task}_{resource}",
                            task=task,
                            task_mode_var=task_mode_var,
                            mode2mapping={
                                mode: self.problem.get_non_renewable_resource_consumption_mapping(
                                    resource=resource, task=task, mode=mode
                                )
                                for mode in self.problem.get_task_modes(task=task)
                            },
                        )
                    )
                else:
                    mode2value = {
                        m: self.problem.get_non_renewable_resource_consumption(
                            resource=resource, task=task, mode=m
                        )
                        for m in self.problem.get_task_modes(task)
                    }
                    mode2var = {
                        m: self.get_task_mode_is_present_variable(task=task, mode=m)
                        for m in self.problem.get_task_modes(task)
                    }
                    self.demands_non_renewable_resource_vars[task, resource] = (
                        create_variable_function_of_mode_on_solver(
                            solver=self,
                            name=f"conso_{task}_{resource}",
                            mode2value=mode2value,
                            mode2var=mode2var,
                            modeling=self.demand_non_renewable_modeling,
                        )
                    )
        self.demands_non_renewable_resource_initialized = True

    def get_non_renewable_resource_demand_variable(
        self, task: Task, resource: NonRenewableResource
    ) -> LinearExprT:
        """Get the variable representing the resource demand by the task.

        Default to a linear expression using consumption per mode and is_present variables.
        If demand variables are indeed created in the cp_model, this should be overriden to return it
        so that non renewable resource constraints are constraining these variables.

        Needed if `self.use_demand_variables_for_non_renewable_resources` is set to True.

        Args:
            task:
            resource:

        Returns:

        """
        if not self.demands_non_renewable_resource_initialized:
            self.initialize_non_renewable_resource_demand_vars()
        return self.demands_non_renewable_resource_vars[task, resource]

    def create_non_renewable_resources_constraint(self, resource: NonRenewableResource):
        """Add the constraint for a non-renewable resource to the cpsat model.

        Constraint ensuring that the total demand on the given resource stay below its capacity.

        """
        self.cp_model.add(
            sum(
                self.get_non_renewable_resource_demand_variable(
                    task=task, resource=resource
                )
                for task in self.problem.tasks_list
            )
            <= self.problem.get_non_renewable_resource_capacity(resource)
        )
