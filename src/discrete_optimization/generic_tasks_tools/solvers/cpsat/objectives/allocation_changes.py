#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Generic

from ortools.sat.python.cp_model import LinearExpr

from discrete_optimization.generic_tasks_tools.allocation import Task, UnaryResource
from discrete_optimization.generic_tasks_tools.objectives.allocation_changes import (
    AllocationSwitchObjectiveComputer,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.auto import (
    GenericSchedulingAutoCpSatSolver,
)

# TODO : investigate, if not better to take into account via :
#  self.problem.get_unary_resource_cost ... !
from discrete_optimization.generic_tasks_tools.solvers.cpsat.objectives.objective_modeler import (
    ObjectiveModelerCpSat,
)


class AllocationSwitchModelerCpSat(ObjectiveModelerCpSat, Generic[Task, UnaryResource]):
    objective_computer: AllocationSwitchObjectiveComputer[Task, UnaryResource]
    solver: GenericSchedulingAutoCpSatSolver
    allocation_change_variables_initialized: bool = False
    allocation_change_variables: dict
    init_value: dict

    def create_allocation_change_variables(self):
        # TODO : Investigate this other way, which is already existing.
        # self.solver.create_allocation_changes_variables()
        # self.solver.add_allocation_changes_constraints(self.objective_computer.base_allocation_solution)
        self.allocation_change_variables = {}
        self.init_value = {}
        model = self.solver.cp_model
        for task in self.solver.problem.tasks_list:
            for unary_resource in self.solver.problem.unary_resources_list:
                self.allocation_change_variables[task, unary_resource] = (
                    model.new_bool_var(name=f"{task}_{unary_resource}_changed")
                )
                self.init_value[task, unary_resource] = (
                    self.objective_computer.base_allocation_solution.is_allocated(
                        task, unary_resource
                    )
                )
                if self.init_value[task, unary_resource] == 1:
                    model.add(
                        self.allocation_change_variables[task, unary_resource]
                        == 1
                        - self.solver.get_task_unary_resource_is_present_variable(
                            task, unary_resource
                        )
                    )
                else:
                    model.add(
                        self.allocation_change_variables[task, unary_resource]
                        == self.solver.get_task_unary_resource_is_present_variable(
                            task, unary_resource
                        )
                    )
                if self.init_value[task, unary_resource]:
                    model.add(
                        self.allocation_change_variables[task, unary_resource] == 1
                    ).only_enforce_if(
                        ~self.solver.get_task_unary_resource_is_present_variable(
                            task, unary_resource
                        )
                    )
                else:
                    model.add(
                        self.allocation_change_variables[task, unary_resource] == 1
                    ).only_enforce_if(
                        self.solver.get_task_unary_resource_is_present_variable(
                            task, unary_resource
                        )
                    )
        self.allocation_change_variables_initialized = True

    def get_objective_expr(self) -> LinearExpr:
        if not self.allocation_change_variables_initialized:
            self.create_allocation_change_variables()
        return sum(
            [
                self.allocation_change_variables[task, unary_resource]
                * (
                    self.objective_computer.get_switch_off_cost(task, unary_resource)
                    if self.init_value[task, unary_resource]
                    else self.objective_computer.get_switch_on_cost(
                        task, unary_resource
                    )
                )
                for task, unary_resource in self.allocation_change_variables
            ]
        )
