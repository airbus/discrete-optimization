#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import re

import didppy as dp

from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.dyn_prog_tools import DpSolver
from discrete_optimization.vrp.problem import VrpSolution
from discrete_optimization.vrp.solvers import VrpSolver
from discrete_optimization.vrp.utils import compute_length_matrix

logger = logging.getLogger(__name__)


class DpVrpSolver(VrpSolver, DpSolver):
    hyperparameters = DpSolver.hyperparameters
    transitions: dict

    def init_model(self, **kwargs):
        """
        DP model for CVRP
        Directly adapted from https://github.com/domain-independent-dp/didp-rs/blob/main/didppy/examples/cvrp.ipynb
        """
        # Number of locations
        n = self.problem.customer_count
        # Number of vehicles
        m = self.problem.vehicle_count
        # Capacity of a vehicle
        q = self.problem.vehicle_capacities[0]
        # Capacities
        capacities = self.problem.vehicle_capacities
        sum_capacities_backward = [
            sum(capacities[i:]) for i in range(self.problem.vehicle_count)
        ] + [0]

        # Weights
        d = [self.problem.customers[i].demand for i in range(n)]
        # Travel cost
        _, c = compute_length_matrix(self.problem)
        c = [[int(10 * c[i, j]) for j in range(c.shape[1])] for i in range(c.shape[0])]
        model = dp.Model()
        customer = model.add_object_type(number=n)
        vehicle_obj = model.add_object_type(number=m)
        # U
        unvisited = model.add_set_var(object_type=customer, target=list(range(1, n)))
        # i
        location = model.add_element_var(object_type=customer, target=0)
        # l
        load = model.add_int_resource_var(target=0, less_is_better=True)
        # k
        vehicles = model.add_int_resource_var(target=0, less_is_better=True)
        vehicles_ = model.add_element_var(object_type=vehicle_obj, target=0)
        weight = model.add_int_table(d)
        distance = model.add_int_table(c)
        capacities = model.add_int_table(self.problem.vehicle_capacities)
        sum_capacities_backward = model.add_int_table(sum_capacities_backward)
        model.add_base_case([unvisited.is_empty(), location == 0])
        self.transitions = {}
        for j in range(1, n):
            visit = dp.Transition(
                name="visit {}".format(j),
                cost=distance[location, j] + dp.IntExpr.state_cost(),
                effects=[
                    (unvisited, unvisited.remove(j)),
                    (location, j),
                    (load, load + weight[j]),
                ],
                preconditions=[
                    unvisited.contains(j),
                    load + weight[j] <= capacities[vehicles_],
                ],
            )
            model.add_transition(visit)
            self.transitions[("visit", "current", j)] = visit

        for j in range(1, n):
            visit_via_depot = dp.Transition(
                name="visit {} with a new vehicle".format(j),
                cost=distance[location, 0] + distance[0, j] + dp.IntExpr.state_cost(),
                effects=[
                    (unvisited, unvisited.remove(j)),
                    (location, j),
                    (load, weight[j]),
                    (vehicles, vehicles + 1),
                    (vehicles_, vehicles_ + 1),
                ],
                preconditions=[unvisited.contains(j), vehicles < m],
            )
            model.add_transition(visit_via_depot)
            self.transitions[("visit", "next_vehicle", j)] = visit_via_depot

        return_to_depot = dp.Transition(
            name="return",
            cost=distance[location, 0] + dp.IntExpr.state_cost(),
            effects=[(location, 0)],
            preconditions=[unvisited.is_empty(), location != 0],
        )
        model.add_transition(return_to_depot)
        self.transitions["return"] = return_to_depot
        model.add_state_constr(
            sum_capacities_backward[vehicles_] - load >= weight[unvisited]
        )
        model.add_state_constr((m - vehicles + 1) * q - load >= weight[unvisited])

        min_distance_to = model.add_int_table(
            [min(c[k][j] for k in range(n) if j != k) for j in range(n)]
        )
        model.add_dual_bound(
            min_distance_to[unvisited]
            + (location != 0).if_then_else(min_distance_to[0], 0)
        )

        min_distance_from = model.add_int_table(
            [min(c[j][k] for k in range(n) if j != k) for j in range(n)]
        )
        model.add_dual_bound(
            min_distance_from[unvisited]
            + (location != 0).if_then_else(min_distance_from[location], 0)
        )
        self.model = model

    def init_model_(self, **kwargs):
        """
        DP model for CVRP
        Directly adapted from https://github.com/domain-independent-dp/didp-rs/blob/main/didppy/examples/cvrp.ipynb
        """
        # Number of locations
        n = self.problem.customer_count
        # Number of vehicles
        m = self.problem.vehicle_count
        # Capacity of a vehicle
        q = self.problem.vehicle_capacities[0]
        # Capacities
        capacities = self.problem.vehicle_capacities
        sum_capacities_backward = [
            sum(capacities[i:]) for i in range(self.problem.vehicle_count)
        ] + [0]

        # Weights
        d = [self.problem.customers[i].demand for i in range(n)]
        # Travel cost
        _, c = compute_length_matrix(self.problem)
        c = [[int(10 * c[i, j]) for j in range(c.shape[1])] for i in range(c.shape[0])]
        model = dp.Model()
        customer = model.add_object_type(number=n)
        vehicle_obj = model.add_object_type(number=m)
        # U
        unvisited = model.add_set_var(
            object_type=customer,
            target=[
                i
                for i in range(self.problem.customer_count)
                if i != self.problem.start_indexes[0]
            ],
        )
        # i
        location = model.add_element_var(
            object_type=customer, target=self.problem.start_indexes[0]
        )
        # l
        load = model.add_int_resource_var(target=0, less_is_better=True)
        # k
        vehicles = model.add_int_resource_var(target=0, less_is_better=True)
        vehicles_ = model.add_element_var(object_type=vehicle_obj, target=0)
        starts_vehicle = model.add_element_table(self.problem.start_indexes)
        ends_vehicle = model.add_element_table(self.problem.end_indexes)
        weight = model.add_int_table(d)
        distance = model.add_int_table(c)
        capacities = model.add_int_table(self.problem.vehicle_capacities)
        sum_capacities_backward = model.add_int_table(sum_capacities_backward)
        model.add_base_case(
            [unvisited.is_empty(), location == self.problem.end_indexes[-1]]
        )
        self.transitions = {}
        for j in range(n):
            visit = dp.Transition(
                name="visit {}".format(j),
                cost=distance[location, j] + dp.IntExpr.state_cost(),
                effects=[
                    (unvisited, unvisited.remove(j)),
                    (location, j),
                    (load, load + weight[j]),
                ],
                preconditions=[
                    unvisited.contains(j),
                    load + weight[j] <= capacities[vehicles_],
                ],
            )
            model.add_transition(visit)
            self.transitions[("visit", "current", j)] = visit
        for j in range(n):
            visit_via_depot = dp.Transition(
                name="visit {} with a new vehicle".format(j),
                cost=distance[location, ends_vehicle[vehicles_]]
                + distance[starts_vehicle[vehicles_ + 1], j]
                + dp.IntExpr.state_cost(),
                effects=[
                    (unvisited, unvisited.remove(j)),
                    (location, j),
                    (load, weight[j]),
                    (vehicles, vehicles + 1),
                    (vehicles_, vehicles_ + 1),
                ],
                preconditions=[unvisited.contains(j), vehicles < m],
            )
            model.add_transition(visit_via_depot)
            self.transitions[("visit", "next_vehicle", j)] = visit_via_depot

        return_to_depot = dp.Transition(
            name="return",
            cost=distance[location, self.problem.end_indexes[-1]]
            + dp.IntExpr.state_cost(),
            effects=[(location, self.problem.end_indexes[-1])],
            preconditions=[unvisited.is_empty(), location != 0],
        )
        model.add_transition(return_to_depot)
        self.transitions["return"] = return_to_depot
        model.add_state_constr(
            sum_capacities_backward[vehicles_] - load >= weight[unvisited]
        )
        model.add_state_constr((m - vehicles + 1) * q - load >= weight[unvisited])

        min_distance_to = model.add_int_table(
            [min(c[k][j] for k in range(n) if j != k) for j in range(n)]
        )
        model.add_dual_bound(
            min_distance_to[unvisited]
            + (location != 0).if_then_else(min_distance_to[0], 0)
        )

        min_distance_from = model.add_int_table(
            [min(c[j][k] for k in range(n) if j != k) for j in range(n)]
        )
        model.add_dual_bound(
            min_distance_from[unvisited]
            + (location != 0).if_then_else(min_distance_from[location], 0)
        )
        self.model = model

    def retrieve_solution(self, sol: dp.Solution) -> Solution:
        list_paths = [[] for _ in range(self.problem.vehicle_count)]

        def extract_visit_number(text):
            match = re.search(r"visit\s(\d+)", text, re.IGNORECASE)
            if match:
                return int(match.group(1))
            return None

        cur_index = 0
        for t in sol.transitions:
            name = t.name
            if "with a new vehicle" in name:
                cur_index += 1
            if "return" in name:
                break
            list_paths[cur_index].append(extract_visit_number(name))
        logger.info(f"Cost: {sol.cost}")
        vrp_sol = VrpSolution(
            problem=self.problem,
            list_start_index=self.problem.start_indexes,
            list_end_index=self.problem.end_indexes,
            list_paths=list_paths,
        )
        return vrp_sol

    def set_warm_start(self, solution: VrpSolution) -> None:
        initial_solution = []
        flattened = []
        for v in range(len(solution.list_paths)):
            p = solution.list_paths[v]
            flattened += p
            if v != len(solution.list_paths) - 1:
                flattened += [("change_vehicle")]
        i = 0
        while i < len(flattened):
            p = flattened[i]
            if p == "change_vehicle":
                initial_solution.append(
                    self.transitions[("visit", "next_vehicle", flattened[i + 1])]
                )
                i += 2
            else:
                initial_solution.append(
                    self.transitions[("visit", "current", flattened[i])]
                )
                i += 1
        initial_solution.append(self.transitions["return"])
        self.initial_solution = initial_solution
