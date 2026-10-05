#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Generic

from ortools.sat.python.cp_model import LinearExprT

from discrete_optimization.generic_tasks_tools.allocation import UnaryResource
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.generic_scheduling import (
    CumulativeResource,
)
from discrete_optimization.generic_tasks_tools.resource_usage_by_unary import (
    ResourceUsageByUnaryResourceProblem,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.allocation import (
    AllocationCpSatSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.cumulative_resource import (
    CumulativeResource,
    CumulativeResourceSchedulingCpSatSolver,
    OtherCalendarResource,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.utils import (
    ModeToValueModeling,
    create_variable_function_of_mode_on_solver,
)


class ResourceUsageByUnaryResourceCpSatSolver(
    AllocationCpSatSolver[Task, UnaryResource],
    CumulativeResourceSchedulingCpSatSolver[
        Task, CumulativeResource, OtherCalendarResource
    ],
    Generic[Task, CumulativeResource, OtherCalendarResource, UnaryResource],
):
    problem: ResourceUsageByUnaryResourceProblem[
        Task, CumulativeResource, UnaryResource
    ]
    unary_dependent_demand_resource_task_initialized: bool = False
    unary_dependent_demands_cumulative_resource_vars: dict[
        Task, dict[CumulativeResource, dict[UnaryResource, LinearExprT]]
    ]
    unary_dependent_demand_cumulative_modeling: ModeToValueModeling = (
        ModeToValueModeling.ENFORCE_IF
    )

    def create_vars_for_unary_dependent_demand(self):
        self.unary_dependent_demands_cumulative_resource_vars = {}
        for task in self.problem.tasks_list:
            nz_ = self.problem.get_non_zero_mode_res_unary(
                task
            )  # mode, resource, unary
            res = set([r[1] for r in nz_])
            if len(res) >= 1:
                self.unary_dependent_demands_cumulative_resource_vars[task] = {}
                for r in res:
                    self.unary_dependent_demands_cumulative_resource_vars[task][r] = {}
                    unary = set([r[2] for r in nz_ if r[1]])
                    for ur in unary:
                        possible_values = {
                            m: self.problem.get_resource_consumption_when_unary_resource_allocated(
                                task=task, mode=m, unary_resource=ur, resource=r
                            )
                            for m in self.problem.get_task_modes(task)
                        }
                        self.unary_dependent_demands_cumulative_resource_vars[task][r][
                            ur
                        ] = create_variable_function_of_mode_on_solver(
                            solver=self,
                            name=f"unary_dependent_{task}_{r}_{ur}",
                            mode2value=possible_values,
                            task=task,
                            modeling=self.unary_dependent_demand_cumulative_modeling,
                            conditional_var=self.get_task_unary_resource_is_present_variable(
                                task=task, unary_resource=ur
                            ),
                        )
        self.unary_dependent_demand_resource_task_initialized = True

    def get_resource_demand_from_unary(
        self, task: Task, resource: CumulativeResource
    ) -> LinearExprT:
        if not self.unary_dependent_demand_resource_task_initialized:
            self.create_vars_for_unary_dependent_demand()
        if task in self.unary_dependent_demands_cumulative_resource_vars:
            if resource in self.unary_dependent_demands_cumulative_resource_vars[task]:
                d = self.unary_dependent_demands_cumulative_resource_vars[task][
                    resource
                ]
                keys = list(d.keys())
                if len(keys) == 1:
                    return d[keys[0]]
                else:
                    return sum([d[k] for k in keys])
        return 0
