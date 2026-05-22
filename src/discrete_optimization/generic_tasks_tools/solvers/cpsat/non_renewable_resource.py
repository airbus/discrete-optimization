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


class NonRenewableCpSatSolver(
    MultimodeCpSatSolver[Task], Generic[Task, NonRenewableResource]
):
    """Base class for cpsat solvers dealing with problem with non-renewable resources."""

    problem: NonRenewableResourceProblem

    def get_non_renewable_resource_demand_variable(
        self, task: Task, resource: NonRenewableResource
    ) -> LinearExprT:
        """Get the variable representing the resource demand by the task.

        Default to a linear expression using consumption per mode and is_present variables.
        If demand variables are indeed created in the cp_model, this should be overriden to return it
        so that non-renewable resource constraints are constraining these variables.

        Args:
            task:
            resource:

        Returns:

        """
        return sum(
            self.get_task_mode_is_present_variable(task=task, mode=mode) * conso
            for mode in self.problem.get_task_modes(task)
            if (
                conso := self.problem.get_non_renewable_resource_consumption(
                    resource=resource, task=task, mode=mode
                )
            )
            > 0
        )

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
