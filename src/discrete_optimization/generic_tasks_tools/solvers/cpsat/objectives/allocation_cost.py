#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from ortools.sat.python.cp_model import LinearExpr

from discrete_optimization.generic_tasks_tools.objectives.allocation_cost import (
    AllocationCostComputer,
    AllocationCostComputerMultimode,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.objectives.objective_modeler import (
    ObjectiveModelerCpSat,
)


class AllocationCostModelerCpSat(ObjectiveModelerCpSat):
    objective_computer: AllocationCostComputer

    def get_objective_expr(self) -> LinearExpr:
        return sum(
            [
                self.solver.get_task_unary_resource_is_present_variable(
                    task=task, unary_resource=unary
                )
                * cost
                for task in self.solver.problem.tasks_list
                for unary in self.solver.problem.unary_resources_list
                if (
                    (
                        cost
                        := self.objective_computer.cost_allocation_resource_to_task(
                            task=task, unary_resource=unary
                        )
                    )
                    != 0
                )
            ]
        )


class AllocationCostMultimodeModelerCpSat(ObjectiveModelerCpSat):
    objective_computer: AllocationCostComputerMultimode

    def get_objective_expr(self) -> LinearExpr:
        for task in self.objective_computer.get_tasks_having_cost():
            self.solver.unary_resource_cost_variables[task] = {}
            for unary_resource in self.solver.problem.unary_resources_list:
                self.solver.unary_resource_cost_variables[task][unary_resource] = (
                    self.solver._create_var_per_mode_if_allocated(
                        name=f"unary_resource_cost_{task}_{unary_resource}",
                        mode2value={
                            mode: self.objective_computer.cost_allocation_resource_to_task_mode(
                                task=task, mode=mode, unary_resource=unary_resource
                            )
                            for mode in self.solver.problem.get_task_modes(task=task)
                        },
                        task=task,
                        unary_resource=unary_resource,
                    )
                )
        return sum(
            self.solver.unary_resource_cost_variables[task][unary_resource]
            for task in self.solver.unary_resource_cost_variables
            for unary_resource in self.solver.unary_resource_cost_variables[task]
        )
