#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from typing import Any

from ortools.sat.python.cp_model import (
    CpSolverSolutionCallback,
    LinearExprT,
)

from discrete_optimization.facility.problem import (
    Customer,
    Facility,
    FacilityProblem,
    FacilitySolution,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat import (
    AllocationCpSatSolver,
)
from discrete_optimization.generic_tools.do_problem import ModeOptim
from discrete_optimization.generic_tools.do_solver import WarmstartMixin

logger = logging.getLogger(__name__)


class CpSatFacilitySolver(AllocationCpSatSolver[Customer, Facility], WarmstartMixin):
    def set_warm_start(self, solution: FacilitySolution) -> None:
        self.cp_model.ClearHints()
        alloc = {
            c: {f: 0 for f in self.problem.facilities} for c in self.problem.customers
        }
        used = set()
        for i in range(self.problem.customer_count):
            f_index = solution.facility_for_customers[i]
            alloc[self.problem.customers[i]][self.problem.facilities[f_index]] = 1
            used.add(self.problem.facilities[f_index])
        for c in alloc:
            for f in alloc[c]:
                self.cp_model.add_hint(self.variables["alloc"][c][f], alloc[c][f])
        for f in self.used_variables:
            if f in used:
                self.cp_model.add_hint(self.used_variables[f], 1)
            else:
                self.cp_model.add_hint(self.used_variables[f], 0)

    problem: FacilityProblem
    variables: dict

    def init_model(self, **kwargs: Any) -> None:
        super().init_model(**kwargs)
        self.variables = {}
        x_alloc = {
            c: {
                f: self.cp_model.NewBoolVar(f"x_{c, f}")
                for f in self.problem.facilities
            }
            for c in self.problem.customers
        }
        for c in self.problem.customers:
            self.cp_model.add_exactly_one([x_alloc[c][f] for f in x_alloc[c]])
        self.variables["alloc"] = x_alloc
        # auto creation of the used variables.
        self.create_used_variables()
        # Capacity constraint
        for f in self.problem.facilities:
            self.cp_model.add(
                sum([x_alloc[c][f] * c.demand for c in x_alloc]) <= f.capacity
            )
        assignment_cost = sum(
            [
                x_alloc[c][f]
                * self.problem.evaluate_customer_facility(facility=f, customer=c)
                for c in x_alloc
                for f in x_alloc[c]
            ]
        )
        setup_cost = sum(
            [self.used_variables[f] * f.setup_cost for f in self.used_variables]
        )
        weights = []
        objs = []
        sense = self.params_objective_function.sense_function
        for obj, obj_weight in zip(
            self.params_objective_function.objectives,
            self.params_objective_function.weights,
        ):
            if sense == ModeOptim.MAXIMIZATION:
                obj_weight = -obj_weight
            if obj == "cost":
                weights.append(obj_weight)
                objs.append(assignment_cost)
            if obj == "setup_cost":
                weights.append(obj_weight)
                objs.append(setup_cost)
        self.cp_model.Minimize(sum([weights[i] * objs[i] for i in range(len(weights))]))

    def get_task_unary_resource_is_present_variable(
        self, task: Customer, unary_resource: Facility
    ) -> LinearExprT:
        return self.variables["alloc"][task][unary_resource]

    def retrieve_solution(
        self, cpsolvercb: CpSolverSolutionCallback
    ) -> FacilitySolution:
        facility_for_customer = [None] * self.problem.customer_count
        for c in self.problem.customers:
            for f in self.variables["alloc"][c]:
                if cpsolvercb.Value(self.variables["alloc"][c][f]):
                    facility_for_customer[c.index] = f.index
        sol = FacilitySolution(
            problem=self.problem, facility_for_customers=facility_for_customer
        )
        fit = self.aggreg_from_sol(sol)
        logger.info(
            f"Fit : {fit}, obj cpsat : {cpsolvercb.objective_value}, bound {cpsolvercb.best_objective_bound}"
        )
        return sol
