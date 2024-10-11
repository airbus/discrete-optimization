#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import re
from enum import Enum
from typing import Any

import didppy as dp

from discrete_optimization.facility.problem import FacilityProblem, FacilitySolution
from discrete_optimization.facility.solvers import FacilitySolver
from discrete_optimization.facility.utils import (
    compute_matrix_distance_facility_problem,
)
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.dyn_prog_tools import DpSolver
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
)

logger = logging.getLogger(__name__)


class DpFacilityModeling(Enum):
    CUSTOMER = 0
    FACILITY = 1


class DpFacilitySolver(DpSolver, FacilitySolver, WarmstartMixin):
    hyperparameters = DpSolver.hyperparameters + [
        EnumHyperparameter(
            "modeling", enum=DpFacilityModeling, default=DpFacilityModeling.CUSTOMER
        )
    ]

    def __init__(self, problem: FacilityProblem, **kwargs: Any):

        super().__init__(problem, **kwargs)
        self.modeling: DpFacilityModeling = None

    def init_model(
        self,
        modeling: DpFacilityModeling = DpFacilityModeling.CUSTOMER,
        **kwargs: Any,
    ) -> None:
        if modeling == DpFacilityModeling.FACILITY:
            self.init_model_factory_view(**kwargs)
            self.modeling = modeling
        if modeling == DpFacilityModeling.CUSTOMER:
            self.init_model_customer_view(**kwargs)
            self.modeling = modeling

    def init_model_customer_view(self, **kwargs: Any) -> None:
        model = dp.Model()
        nb_facilities = self.problem.facility_count
        nb_customers = self.problem.customer_count
        capacities_facilities = [f.capacity for f in self.problem.facilities]
        setup_cost = [10 * int(f.setup_cost) for f in self.problem.facilities]
        matrix = compute_matrix_distance_facility_problem(self.problem)
        matrix = [
            [int(10 * matrix[i, j]) for j in range(matrix.shape[1])]
            for i in range(matrix.shape[0])
        ]
        distance_matrix = model.add_int_table(matrix)
        capacities = model.add_int_table(capacities_facilities)
        setup_cost = model.add_int_table(setup_cost)
        customer = model.add_object_type(number=nb_customers)
        factories = model.add_object_type(number=nb_facilities)
        unallocated = model.add_set_var(
            object_type=customer, target=range(nb_customers)
        )
        used_facilities = model.add_set_var(object_type=factories, target=[])
        current_facility = model.add_element_var(object_type=factories, target=0)
        # nb_facility_used = model.add_int_resource_var(target=0, less_is_better=True)
        current_capacity = model.add_int_resource_var(target=0)
        model.add_base_case([unallocated.is_empty()])
        self.transitions = {}
        for i in range(nb_customers):
            transition = dp.Transition(
                name=f"alloc_{i}",
                cost=distance_matrix[i, current_facility] + dp.IntExpr.state_cost(),
                effects=[
                    (unallocated, unallocated.remove(i)),
                    (
                        current_capacity,
                        current_capacity + self.problem.customers[i].demand,
                    ),
                ],
                preconditions=[
                    unallocated.contains(i),
                    ~used_facilities.is_empty(),
                    current_capacity + self.problem.customers[i].demand
                    <= capacities[current_facility],
                ],
            )
            model.add_transition(transition)
            self.transitions[("current", "customer", i)] = transition
            for f in range(nb_facilities):
                transition = dp.Transition(
                    name=f"alloc_via_{f}_{i}",
                    cost=setup_cost[f]
                    + distance_matrix[i, f]
                    + dp.IntExpr.state_cost(),
                    effects=[
                        (unallocated, unallocated.remove(i)),
                        (used_facilities, used_facilities.add(f)),
                        (current_facility, f),
                        (current_capacity, self.problem.customers[i].demand),
                    ],
                    preconditions=[
                        unallocated.contains(i),
                        ~(used_facilities.contains(f)),
                        self.problem.customers[i].demand <= capacities[f],
                    ],
                )
                model.add_transition(transition)
                self.transitions[(f, "customer", i)] = transition
        self.model = model

    def init_model_factory_view(self, **kwargs: Any) -> None:
        model = dp.Model()
        nb_facilities = self.problem.facility_count
        nb_customers = self.problem.customer_count
        capacities_facilities = [f.capacity for f in self.problem.facilities]
        setup_cost = [10 * int(f.setup_cost) for f in self.problem.facilities]
        matrix = compute_matrix_distance_facility_problem(self.problem)
        matrix = [
            [int(10 * matrix[i, j]) for j in range(matrix.shape[1])]
            for i in range(matrix.shape[0])
        ]
        distance_matrix = model.add_int_table(matrix)
        capacities = model.add_int_table(capacities_facilities)
        setup_cost = model.add_int_table(setup_cost)
        customer = model.add_object_type(number=nb_customers)
        factories = model.add_object_type(number=nb_facilities)
        used_facilities = model.add_set_var(object_type=factories, target=[])
        demands = model.add_int_table([f.demand for f in self.problem.customers])
        used_capacities = [
            model.add_int_resource_var(target=0, less_is_better=False)
            for f in range(nb_facilities)
        ]
        current_customer = model.add_element_var(object_type=customer, target=0)
        self.transitions = {}
        for f in range(nb_facilities):
            new_cost = (used_facilities.contains(f)).if_then_else(0, setup_cost[f])
            alloc = dp.Transition(
                name=f"alloc_to_{f}",
                cost=new_cost
                + distance_matrix[current_customer, f]
                + dp.IntExpr.state_cost(),
                effects=[
                    (used_facilities, used_facilities.add(f)),
                    (
                        used_capacities[f],
                        used_capacities[f] + demands[current_customer],
                    ),
                    (current_customer, current_customer + 1),
                ],
                preconditions=[
                    used_capacities[f] + demands[current_customer] <= capacities[f],
                    current_customer < self.problem.customer_count,
                ],
            )
            model.add_transition(alloc)
            self.transitions[f] = alloc
        model.add_base_case([current_customer == self.problem.customer_count])
        # min_distance_to = model.add_int_table(
        #     [
        #         min(matrix[k][j] for j in range(self.problem.facility_count))
        #         for k in range(self.problem.customer_count)
        #     ]
        # )
        # model.add_dual_bound(sum(min_distance_to))
        self.model = model

    def retrieve_solution(self, sol: dp.Solution) -> Solution:
        if self.modeling == DpFacilityModeling.FACILITY:
            return self.retrieve_solution_factory_view(sol)
        if self.modeling == DpFacilityModeling.CUSTOMER:
            return self.retrieve_solution_customer_view(sol)

    def retrieve_solution_customer_view(self, sol: dp.Solution) -> Solution:
        def extract_ints(word):
            return tuple(int(num) for num in re.findall(r"\d+", word))

        solution = FacilitySolution(
            problem=self.problem,
            facility_for_customers=[None for i in range(self.problem.customer_count)],
        )
        current_factory = 0
        for t in sol.transitions:
            try:
                if "alloc_via" in t.name:
                    factory, customer = extract_ints(t.name)
                    current_factory = factory
                else:
                    customer = extract_ints(t.name)[0]
                solution.facility_for_customers[customer] = current_factory
            except:
                pass
        return solution

    def retrieve_solution_factory_view(self, sol: dp.Solution) -> Solution:
        def extract_ints(word):
            return tuple(int(num) for num in re.findall(r"\d+", word))

        solution = FacilitySolution(
            problem=self.problem,
            facility_for_customers=[None for i in range(self.problem.customer_count)],
        )
        current_factory = 0
        customer = 0

        for t in sol.transitions:
            current_factory = extract_ints(t.name)[0]
            solution.facility_for_customers[customer] = current_factory
            customer += 1
        return solution

    def set_warm_start(self, solution: FacilitySolution) -> None:
        if self.modeling == DpFacilityModeling.FACILITY:
            self.initial_solution = [
                self.transitions[f] for f in solution.facility_for_customers
            ]
        if self.modeling == DpFacilityModeling.CUSTOMER:
            initial_solution = []
            customer_by_facilities = {}
            for i in range(len(solution.facility_for_customers)):
                f = solution.facility_for_customers[i]
                if f not in customer_by_facilities:
                    customer_by_facilities[f] = []
                customer_by_facilities[f].append(i)
            for f in sorted(customer_by_facilities):
                list_customer = customer_by_facilities[f]
                initial_solution.append(
                    self.transitions[(f, "customer", list_customer[0])]
                )
                for customer in list_customer[1:]:
                    initial_solution.append(
                        self.transitions[("current", "customer", customer)]
                    )
            self.initial_solution = initial_solution
