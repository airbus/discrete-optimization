#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import re
from typing import Any, List, Optional

import didppy as dp

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_solver import ResultStorage
from discrete_optimization.generic_tools.dyn_prog_tools import DidSolver
from discrete_optimization.vrp.solver.vrp_solver import SolverVrp
from discrete_optimization.vrp.vrp_model import VrpSolution, trivial_solution
from discrete_optimization.vrp.vrp_toolbox import compute_length_matrix


class DidVrpSolver(SolverVrp, DidSolver):
    hyperparameters = DidSolver.hyperparameters

    def init_model(self, **kwargs):
        """Directly adapted from https://didppy.readthedocs.io/en/stable/tutorial.html"""
        # Number of locations
        n = self.problem.customer_count
        # Number of vehicles
        m = self.problem.vehicle_count
        # Capacity of a vehicle
        q = self.problem.vehicle_capacities[0]
        # Weights
        d = [self.problem.customers[i].demand for i in range(n)]
        # Travel cost
        c, _ = compute_length_matrix(self.problem)
        model = dp.Model()
        customer = model.add_object_type(number=n)
        # U
        unvisited = model.add_set_var(object_type=customer, target=list(range(1, n)))
        # i
        location = model.add_element_var(object_type=customer, target=0)
        # l
        load = model.add_int_resource_var(target=0, less_is_better=True)
        # k
        vehicles = model.add_int_resource_var(target=1, less_is_better=True)
        weight = model.add_int_table(d)
        distance = model.add_int_table(c)
        model.add_base_case([unvisited.is_empty(), location == 0])
        for j in range(1, n):
            visit = dp.Transition(
                name="visit {}".format(j),
                cost=distance[location, j] + dp.IntExpr.state_cost(),
                effects=[
                    (unvisited, unvisited.remove(j)),
                    (location, j),
                    (load, load + weight[j]),
                ],
                preconditions=[unvisited.contains(j), load + weight[j] <= q],
            )
            model.add_transition(visit)

        for j in range(1, n):
            visit_via_depot = dp.Transition(
                name="visit {} with a new vehicle".format(j),
                cost=distance[location, 0] + distance[0, j] + dp.IntExpr.state_cost(),
                effects=[
                    (unvisited, unvisited.remove(j)),
                    (location, j),
                    (load, weight[j]),
                    (vehicles, vehicles + 1),
                ],
                preconditions=[unvisited.contains(j), vehicles < m],
            )
            model.add_transition(visit_via_depot)

        return_to_depot = dp.Transition(
            name="return",
            cost=distance[location, 0] + dp.IntExpr.state_cost(),
            effects=[(location, 0)],
            preconditions=[unvisited.is_empty(), location != 0],
        )
        model.add_transition(return_to_depot)

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

    def solve(
        self, callbacks: Optional[List[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        solver_cls = kwargs["solver"]
        solver = solver_cls(self.model, time_limit=10, quiet=False)
        solution = solver.search()
        print("Transitions to apply:")
        print("")
        sol = VrpSolution(
            problem=self.problem,
            list_start_index=self.problem.start_indexes,
            list_end_index=self.problem.end_indexes,
            list_paths=[],
        )
        list_paths = [[] for i in range(self.problem.vehicle_count)]

        def extract_visit_number(text):
            match = re.search(r"visit\s(\d+)", text, re.IGNORECASE)
            if match:
                return int(match.group(1))
            return None

        cur_index = 0
        for t in solution.transitions:
            name = t.name
            if "with a new vehicle" in name:
                cur_index += 1
            if "return" in name:
                break
            list_paths[cur_index].append(extract_visit_number(name))
            print(name)
            print(extract_visit_number(name))
        print("")
        print("Cost: {}".format(solution.cost))
        sol = VrpSolution(
            problem=self.problem,
            list_start_index=self.problem.start_indexes,
            list_end_index=self.problem.end_indexes,
            list_paths=list_paths,
        )
        fit = self.aggreg_from_sol(sol)
        return self.create_result_storage([(sol, fit)])
