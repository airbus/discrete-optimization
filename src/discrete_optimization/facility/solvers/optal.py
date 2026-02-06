#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from enum import Enum
from typing import Any

import optalcp as cp

from discrete_optimization.facility.problem import (
    Customer,
    Facility,
    FacilityProblem,
    FacilitySolution,
)
from discrete_optimization.facility.utils import (
    compute_matrix_distance_facility_problem,
)
from discrete_optimization.generic_tasks_tools.allocation import UnaryResource
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.solvers.optalcp_tasks_solver import (
    AllocationOptalSolver,
    SchedulingOptalSolver,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
)


class ModelingOptal(Enum):
    ALLOCATION = "Allocation"
    SCHEDULING = "Scheduling"


class OptalFacilitySolver(
    AllocationOptalSolver[Facility, Customer],
    SchedulingOptalSolver[Customer],
):
    hyperparameters = [
        EnumHyperparameter(
            name="modeling", enum=ModelingOptal, default=ModelingOptal.ALLOCATION
        )
    ]
    problem: FacilityProblem

    def __init__(
        self,
        problem: FacilityProblem,
        params_objective_function: ParamsObjectiveFunction | None = None,
        **kwargs,
    ) -> None:
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}
        self.modeling = None

    def init_model(self, **kwargs: Any) -> None:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        if kwargs["modeling"] == ModelingOptal.SCHEDULING:
            self.init_scheduling()
            self.modeling = ModelingOptal.SCHEDULING
        if kwargs["modeling"] == ModelingOptal.ALLOCATION:
            self.init_allocation()
            self.modeling = ModelingOptal.ALLOCATION

    def init_allocation(self) -> None:
        self.cp_model = cp.Model()
        allocation = {}
        nb_facilities = self.problem.facility_count
        matrix = compute_matrix_distance_facility_problem(self.problem)
        cost = {}
        for t in self.problem.tasks_list:
            for c in self.problem.unary_resources_list:
                allocation[(t, c)] = self.cp_model.bool_var(
                    name=f"allocation_customer_{t}_{c}"
                )
            self.cp_model.enforce(
                self.cp_model.sum(
                    [allocation[(t, c)] for c in self.problem.unary_resources_list]
                )
                == 1
            )
            cost[t] = self.cp_model.sum(
                [
                    int(matrix[self.problem.customers.index(t), i])
                    * allocation[t, self.problem.unary_resources_list[i]]
                    for i in range(self.problem.facility_count)
                ]
            )
        for c in self.problem.unary_resources_list:
            self.cp_model.enforce(
                self.cp_model.sum(
                    [allocation[(t, c)] * t.demand for t in self.problem.tasks_list]
                )
                <= c.capacity
            )
        used_facility = {
            i: self.cp_model.max(
                [
                    allocation[t, self.problem.unary_resources_list[i]]
                    for t in self.problem.tasks_list
                ]
            )
            for i in range(self.problem.facility_count)
        }
        cost_setup = self.cp_model.sum(
            [
                used_facility[i] * int(self.problem.facilities[i].setup_cost)
                for i in range(self.problem.facility_count)
            ]
        )
        cost_alloc = self.cp_model.sum([cost[t] for t in cost])
        self.variables["allocation"] = allocation
        self.cp_model.minimize(cost_setup + cost_alloc)

    def init_scheduling(self):
        self.cp_model = cp.Model()
        intervals = {}
        allocation = {}
        nb_facilities = self.problem.facility_count
        matrix = compute_matrix_distance_facility_problem(self.problem)
        cost_function = {}
        cost = {}
        for t in self.problem.tasks_list:
            intervals[t] = self.cp_model.interval_var(
                start=(0, nb_facilities - 1),
                end=(1, nb_facilities),
                length=1,
                name=f"interval_customer_{t}",
            )
            cost_function[t] = self.cp_model.step_function(
                [
                    (i, int(matrix[self.problem.customers.index(t), i]))
                    for i in range(matrix.shape[1])
                ]
            )
            cost[t] = self.cp_model.eval(
                cost_function[t], self.cp_model.start(intervals[t])
            )
        max_capa_facilities = max([f.capacity for f in self.problem.facilities])
        self.cp_model.enforce(
            self.cp_model.sum(
                [self.cp_model.pulse(intervals[t], t.demand) for t in intervals]
                + [
                    self.cp_model.pulse(
                        self.cp_model.interval_var(start=i, end=i + 1),
                        max_capa_facilities - self.problem.facilities[i].capacity,
                    )
                    for i in range(self.problem.facility_count)
                ]
            )
            <= max_capa_facilities
        )
        self.variables["intervals"] = intervals
        used_factory = {
            i: self.cp_model.max(
                [self.cp_model.start(intervals[t]) == i for t in intervals]
            )
            for i in range(self.problem.facility_count)
        }
        cost_setup = self.cp_model.sum(
            [
                used_factory[i] * int(self.problem.facilities[i].setup_cost)
                for i in range(self.problem.facility_count)
            ]
        )
        cost_alloc = self.cp_model.sum([cost[t] for t in cost])
        self.cp_model.minimize(cost_setup + cost_alloc)

    def get_task_unary_resource_is_present_variable(
        self, task: Task, unary_resource: UnaryResource
    ) -> cp.BoolExpr:
        index = self.problem.get_index_from_unary_resource(unary_resource)
        return (
            self.get_task_start_or_end_variable(task, start_or_end=StartOrEnd.START)
            == index
        )

    def get_task_interval_variable(self, task: Task) -> cp.IntervalVar:
        return self.variables["intervals"][task]

    def retrieve_solution(self, result: cp.SolveResult) -> Solution:
        if self.modeling == ModelingOptal.SCHEDULING:
            allocation = [
                int(result.solution.get_start(self.get_task_interval_variable(t)))
                for t in self.problem.tasks_list
            ]
            return FacilitySolution(
                problem=self.problem, facility_for_customers=allocation
            )
        elif self.modeling == ModelingOptal.ALLOCATION:
            allocation = []
            for i in range(len(self.problem.tasks_list)):
                t = self.problem.tasks_list[i]
                for c in self.problem.unary_resources_list:
                    if result.solution.get_value(self.variables["allocation"][(t, c)]):
                        allocation.append(self.problem.get_index_from_unary_resource(c))
            return FacilitySolution(
                problem=self.problem, facility_for_customers=allocation
            )
