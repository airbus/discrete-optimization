#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Generic

from ortools.linear_solver.python.model_builder import LinearExprT
from ortools.sat.python.cp_model import Domain

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
    demands_non_renewable_resource_initialized: bool = False
    demands_non_renewable_resource_vars: dict[
        tuple[Task, NonRenewableResource], LinearExprT
    ]

    def initialize_non_renewable_resource_demand_vars(
        self, use_enforce_if_instead_of_sum: bool = True
    ):
        """
        Build either expression or variable array for resource demand.
        For task for which resource demand only depends on its own mode, this is a simple expression,
        While for dependent consumption based of other task mode, additional variable is added.
        """
        self.demands_non_renewable_resource_vars = {}
        for task in self.problem.tasks_list:
            for resource in self.problem.non_renewable_resources_list:
                if self.problem.is_non_renewable_resource_task_consumption_dependent(
                    resource=resource, task=task
                ):
                    possible_values = self.problem.get_possible_non_renewable_resource_consumption_all_modes(
                        task=task, resource=resource
                    )
                    self.demands_non_renewable_resource_vars[task, resource] = (
                        self.cp_model.new_int_var_from_domain(
                            domain=Domain.FromValues(list(possible_values)),
                            name=f"conso_{task}_{resource}",
                        )
                    )
                    for mode in self.problem.get_task_modes(task=task):
                        mapping = (
                            self.problem.get_non_renewable_resource_consumption_mapping(
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
                                    self.demands_non_renewable_resource_vars[
                                        task, resource
                                    ]
                                    == value
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
                    if not use_enforce_if_instead_of_sum:
                        self.demands_non_renewable_resource_vars[task, resource] = sum(
                            self.get_task_mode_is_present_variable(task=task, mode=mode)
                            * conso
                            for mode in self.problem.get_task_modes(task)
                            if (
                                conso
                                := self.problem.get_non_renewable_resource_consumption(
                                    resource=resource, task=task, mode=mode
                                )
                            )
                            > 0
                        )
                    else:
                        possible_values = self.problem.get_possible_non_renewable_resource_consumption_all_modes(
                            resource=resource, task=task
                        )
                        self.demands_non_renewable_resource_vars[task, resource] = (
                            self.cp_model.new_int_var_from_domain(
                                domain=Domain.FromValues(list(possible_values)),
                                name=f"conso_{task}_{resource}",
                            )
                        )
                        for mode in self.problem.get_task_modes(task=task):
                            value = self.problem.get_non_renewable_resource_consumption(
                                resource=resource, task=task, mode=mode
                            )
                            self.cp_model.add(
                                self.demands_non_renewable_resource_vars[task, resource]
                                == value
                            ).only_enforce_if(
                                self.get_task_mode_is_present_variable(
                                    task=task, mode=mode
                                )
                            )
        self.demands_non_renewable_resource_initialized = True

    def get_non_renewable_resource_demand_variable(
        self, task: Task, resource: NonRenewableResource
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
