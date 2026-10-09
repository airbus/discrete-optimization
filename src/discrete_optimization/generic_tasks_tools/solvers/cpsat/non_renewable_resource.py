#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Generic

from ortools.linear_solver.python.model_builder import LinearExprT

from discrete_optimization.generic_tasks_tools.allocation import UnaryResource
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.non_renewable_resource import (
    NonRenewableResource,
    NonRenewableResourceProblem,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.allocation import (
    MultimodeAllocationCpSatSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.utils import (
    ModeToValueModeling,
    create_resource_dependent_variable,
    create_variable_function_of_mode_on_solver,
)


class NonRenewableCpSatSolver(
    MultimodeAllocationCpSatSolver[Task, UnaryResource],
    Generic[Task, NonRenewableResource, UnaryResource],
):
    """Base class for cpsat solvers dealing with problem with non-renewable resources."""

    problem: NonRenewableResourceProblem
    # Mode dependent
    mode_defined_nr_resource_initialized: bool = False
    mode_defined_non_renewable_resource_vars: dict[
        tuple[Task, NonRenewableResource], LinearExprT
    ]
    mode_defined_non_renewable_modeling: ModeToValueModeling = (
        ModeToValueModeling.ENFORCE_IF
    )

    # Unary resource dependent
    unary_resource_defined_nr_resource_initialized: bool = False
    unary_dependent_demands_nr_resource_vars: dict[
        Task, dict[NonRenewableResource, dict[UnaryResource, LinearExprT]]
    ]
    unary_dependent_demand_nr_modeling: ModeToValueModeling = (
        ModeToValueModeling.ENFORCE_IF
    )

    def initialize_mode_defined_non_renewable_resource_demand_vars(self):
        """
        Build either expression or variable array for resource demand.
        For task for which resource demand only depends on its own mode, this is a simple expression,
        While for dependent consumption based of other task mode, additional variable is added.
        """
        self.mode_defined_non_renewable_resource_vars = {}
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
                    self.mode_defined_non_renewable_resource_vars[task, resource] = (
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
                    self.mode_defined_non_renewable_resource_vars[task, resource] = (
                        create_variable_function_of_mode_on_solver(
                            solver=self,
                            name=f"conso_{task}_{resource}",
                            mode2value=mode2value,
                            task=task,
                            modeling=self.mode_defined_non_renewable_modeling,
                        )
                    )
        self.mode_defined_nr_resource_initialized = True

    def create_vars_for_unary_dependent_nr_demand(self):
        self.unary_dependent_demands_nr_resource_vars = {}
        for task in self.problem.tasks_list:
            nz_ = self.problem.get_non_zero_mode_nr_res_unary(
                task
            )  # mode, resource, unary
            res = set([r[1] for r in nz_])
            if len(res) >= 1:
                self.unary_dependent_demands_nr_resource_vars[task] = {}
                for r in res:
                    self.unary_dependent_demands_nr_resource_vars[task][r] = {}
                    unary = set([r[2] for r in nz_ if r[1]])
                    for ur in unary:
                        possible_values = {
                            m: self.problem.get_nr_resource_consumption_when_unary_resource_allocated(
                                task=task, mode=m, unary_resource=ur, resource=r
                            )
                            for m in self.problem.get_task_modes(task)
                        }
                        self.unary_dependent_demands_nr_resource_vars[task][r][ur] = (
                            create_variable_function_of_mode_on_solver(
                                solver=self,
                                name=f"unary_dependent_{task}_{r}_{ur}",
                                mode2value=possible_values,
                                task=task,
                                modeling=self.unary_dependent_demand_nr_modeling,
                                conditional_var=self.get_task_unary_resource_is_present_variable(
                                    task=task, unary_resource=ur
                                ),
                            )
                        )
        self.unary_resource_defined_nr_resource_initialized = True

    def get_nr_resource_demand_from_unary(
        self, task: Task, resource: NonRenewableResource
    ) -> LinearExprT:
        if not self.unary_resource_defined_nr_resource_initialized:
            self.create_vars_for_unary_dependent_nr_demand()
        if task in self.unary_dependent_demands_nr_resource_vars:
            if resource in self.unary_dependent_demands_nr_resource_vars[task]:
                d = self.unary_dependent_demands_nr_resource_vars[task][resource]
                keys = list(d.keys())
                if len(keys) == 1:
                    return d[keys[0]]
                else:
                    return sum([d[k] for k in keys])
        return 0

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
        if not self.mode_defined_nr_resource_initialized:
            self.initialize_mode_defined_non_renewable_resource_demand_vars()

        return self.mode_defined_non_renewable_resource_vars[
            task, resource
        ] + self.get_nr_resource_demand_from_unary(task=task, resource=resource)

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
