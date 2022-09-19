#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Optional

import numpy as np

from discrete_optimization.facility.facility_model import (
    FacilityProblem,
    FacilitySolution,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.do_solver import ResultStorage, SolverDO


class GreedySolverFacility(SolverDO):
    """
    build a trivial solution
    pack the facilities one by one until all the customers are served
    """

    def __init__(
        self,
        facility_problem: FacilityProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
    ):
        self.facility_problem = facility_problem
        (
            self.aggreg_sol,
            self.aggreg_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.facility_problem,
            params_objective_function=params_objective_function,
        )

    def solve(self, **kwargs) -> ResultStorage:
        solution = [-1] * self.facility_problem.customer_count
        capacity_remaining = [f.capacity for f in self.facility_problem.facilities]
        facility_index = 0
        for index in range(len(self.facility_problem.customers)):
            customer = self.facility_problem.customers[index]
            if capacity_remaining[facility_index] >= customer.demand:
                solution[index] = facility_index
                capacity_remaining[facility_index] -= customer.demand
            else:
                facility_index += 1
                assert capacity_remaining[facility_index] >= customer.demand
                solution[index] = facility_index
                capacity_remaining[facility_index] -= customer.demand
        sol = FacilitySolution(
            problem=self.facility_problem, facility_for_customers=solution
        )
        fit = self.aggreg_sol(sol)
        return ResultStorage(
            list_solution_fits=[(sol, fit)],
            best_solution=sol,
            mode_optim=self.params_objective_function.sense_function,
        )


class GreedySolverDistanceBased(SolverDO):
    def __init__(
        self,
        facility_problem: FacilityProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
    ):
        self.facility_problem = facility_problem
        (
            self.aggreg_sol,
            self.aggreg_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.facility_problem,
            params_objective_function=params_objective_function,
        )
        self.matrix_cost = np.zeros(
            (self.facility_problem.customer_count, self.facility_problem.facility_count)
        )
        for k in range(self.facility_problem.customer_count):
            for j in range(self.facility_problem.facility_count):
                self.matrix_cost[
                    k, j
                ] = self.facility_problem.evaluate_customer_facility(
                    facility=self.facility_problem.facilities[j],
                    customer=self.facility_problem.customers[k],
                )
        self.min_distance = np.min(self.matrix_cost, axis=1)
        self.sorted_distance = np.argsort(self.matrix_cost, axis=1)
        self.sorted_customers = np.argsort(-self.min_distance)
        # sort the customers based on the minimum distance of facility (so the first element is the one which is the furthest to a facility
        self.available_demands = np.array(
            [
                self.facility_problem.facilities[k].capacity
                for k in range(self.facility_problem.facility_count)
            ]
        )

    def solve(self, **kwargs):
        solution = [-1] * self.facility_problem.customer_count
        capacity_remaining = np.copy(self.available_demands)
        for customer in self.sorted_customers:
            prior_customers = kwargs.get("prio", {}).get(
                customer, self.sorted_distance[customer, :]
            )
            if any(x % 1 != x for x in prior_customers):
                prior_customers = self.sorted_distance[customer, :]
            for f in prior_customers:
                f = int(f)
                if (
                    capacity_remaining[f]
                    >= self.facility_problem.customers[customer].demand
                ):
                    solution[customer] = f
                    capacity_remaining[f] -= self.facility_problem.customers[
                        customer
                    ].demand
                    break
        sol = FacilitySolution(
            problem=self.facility_problem, facility_for_customers=solution
        )
        fit = self.aggreg_sol(sol)
        return ResultStorage(
            list_solution_fits=[(sol, fit)],
            best_solution=sol,
            mode_optim=self.params_objective_function.sense_function,
        )
