#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
# Adapted from https://github.com/domain-independent-dp/didp-rs/blob/main/didppy/examples/tsptw.ipynb
from typing import Any

import didppy as dp

from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.dyn_prog_tools import DpSolver
from discrete_optimization.tsptw.problem import TSPTWProblem, TSPTWSolution


class DpTspTwSolver(DpSolver):

    problem: TSPTWProblem
    transitions: dict

    def init_model(self, **kwargs: Any) -> None:
        self.transitions = {}
        model = dp.Model(float_cost=True)
        n = self.problem.nb_nodes
        depot = self.problem.depot_node
        release = [x[0] for x in self.problem.time_windows]
        deadline = [x[1] for x in self.problem.time_windows]
        customer = model.add_object_type(number=n)
        # U
        unvisited = model.add_set_var(
            object_type=customer, target=self.problem.customers
        )
        # i
        location = model.add_element_var(object_type=customer, target=depot)
        # t
        time = model.add_float_resource_var(target=0, less_is_better=True)

        travel_time = model.add_float_table(self.problem.distance_matrix)
        transition_time = [
            model.add_float_state_fun(
                dp.max(time + travel_time[location, j], release[j]),
                name=f"delta_time_{j}",
            )
            for j in range(self.problem.nb_nodes)
        ]
        for j in self.problem.customers:
            visit = dp.Transition(
                name=f"visit_{j}",
                cost=transition_time[j] - time + dp.FloatExpr.state_cost(),
                preconditions=[unvisited.contains(j)],
                effects=[
                    (unvisited, unvisited.remove(j)),
                    (location, j),
                    (time, transition_time[j]),
                ],
            )
            model.add_transition(visit)
            self.transitions[f"visit_{j}"] = {"transition": visit, "customer": j}

        return_to_depot = dp.Transition(
            name="return",
            cost=travel_time[location, self.problem.depot_node]
            + dp.FloatExpr.state_cost(),
            effects=[
                (location, self.problem.depot_node),
                (time, time + travel_time[location, self.problem.depot_node]),
            ],
            preconditions=[unvisited.is_empty(), location != self.problem.depot_node],
        )
        self.transitions["return"] = {
            "transition": return_to_depot,
            "customer": self.problem.depot_node,
        }
        model.add_transition(return_to_depot)

        model.add_base_case([unvisited.is_empty(), location == self.problem.depot_node])

        for j in self.problem.customers:
            model.add_state_constr(
                ~unvisited.contains(j)
                | (time + travel_time[location, j] <= deadline[j])
            )
        min_to = model.add_float_table(
            [
                min(self.problem.distance_matrix[k, j] for k in range(n) if k != j)
                for j in range(n)
            ]
        )

        model.add_dual_bound(
            min_to[unvisited]
            + (location != self.problem.depot_node).if_then_else(
                min_to[self.problem.depot_node], 0
            )
        )

        min_from = model.add_float_table(
            [
                min(self.problem.distance_matrix[j, k] for k in range(n) if k != j)
                for j in range(n)
            ]
        )

        model.add_dual_bound(
            min_from[unvisited]
            + (location != self.problem.depot_node).if_then_else(min_from[location], 0)
        )
        self.model = model

    def retrieve_solution(self, sol: dp.Solution) -> TSPTWSolution:
        path = []
        for t in sol.transitions:
            path.append(self.transitions[t.name]["customer"])
        return TSPTWSolution(problem=self.problem, permutation=path[:-1])
