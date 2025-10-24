#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Dict, List, Optional, Tuple

import numpy as np

from discrete_optimization.generic_tools.do_problem import (
    EncodingRegister,
    ModeOptim,
    ObjectiveDoc,
    ObjectiveHandling,
    ObjectiveRegister,
    Problem,
    Solution,
    TypeAttribute,
    TypeObjective,
)


class TSPTWSolution(Solution):
    """
    Solution class for the TSP-TW problem.

    Attributes:
        problem (TSPTWProblem): The problem instance.
        permutation (List[int]): A list of customer node indices in the order they are visited.
                                 The depot is not included in this list.
        arrival_times (Dict[int, float]): A dictionary mapping each node index to its arrival time.
        start_service_times (Dict[int, float]): A dictionary mapping each node to the time service begins.
        makespan (float): The total time of the tour, from leaving the depot to returning. This is the primary objective.
        tw_violation (float): The total violation of time windows (sum of lateness at each node).
    """

    def __init__(
        self,
        problem: "TSPTWProblem",
        permutation: List[int],
        arrival_times: Optional[Dict[int, float]] = None,
        start_service_times: Optional[Dict[int, float]] = None,
        makespan: Optional[float] = None,
        tw_violation: Optional[float] = None,
    ):
        self.problem = problem
        self.permutation = permutation
        self.arrival_times = arrival_times if arrival_times is not None else {}
        self.start_service_times = (
            start_service_times if start_service_times is not None else {}
        )
        self.makespan = makespan
        self.tw_violation = tw_violation

    def copy(self) -> "TSPTWSolution":
        return TSPTWSolution(
            problem=self.problem,
            permutation=list(self.permutation),
            arrival_times=self.arrival_times.copy(),
            start_service_times=self.start_service_times.copy(),
            makespan=self.makespan,
            tw_violation=self.tw_violation,
        )

    def lazy_copy(self) -> "TSPTWSolution":
        return TSPTWSolution(
            problem=self.problem,
            permutation=self.permutation,
            arrival_times=self.arrival_times,
            start_service_times=self.start_service_times,
            makespan=self.makespan,
            tw_violation=self.tw_violation,
        )

    def change_problem(self, new_problem: Problem) -> None:
        if not isinstance(new_problem, TSPTWProblem):
            raise ValueError("new_problem must be a TSPTWProblem instance.")
        self.problem = new_problem
        # Invalidate evaluated metrics as they depend on the problem
        self.arrival_times = {}
        self.start_service_times = {}
        self.makespan = None
        self.tw_violation = None

    def __str__(self) -> str:
        path_str = " -> ".join(
            map(
                str,
                [self.problem.depot_node]
                + self.permutation
                + [self.problem.depot_node],
            )
        )
        return (
            f"Path: {path_str}\n"
            f"Makespan: {self.makespan:.2f}\n"
            f"Time Window Violation: {self.tw_violation:.2f}"
        )


class TSPTWProblem(Problem):
    """
    Traveling Salesman Problem with Time Windows (TSP-TW) Problem class.
    """

    def __init__(
        self,
        nb_nodes: int,
        distance_matrix: np.ndarray,
        time_windows: List[Tuple[int, int]],
        depot_node: int = 0,
    ):
        self.nb_nodes = nb_nodes
        self.distance_matrix = distance_matrix
        self.time_windows = time_windows
        self.depot_node = depot_node
        self.customers = sorted(
            [i for i in range(self.nb_nodes) if i != self.depot_node]
        )
        self.nb_customers = len(self.customers)

    def get_attribute_register(self) -> EncodingRegister:
        return EncodingRegister(
            {
                "permutation": {
                    "name": "permutation",
                    "type": [TypeAttribute.PERMUTATION],
                    "n": self.nb_customers,
                    "arr": self.customers,
                }
            }
        )

    def get_objective_register(self) -> ObjectiveRegister:
        return ObjectiveRegister(
            objective_sense=ModeOptim.MINIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc={
                "makespan": ObjectiveDoc(
                    type=TypeObjective.OBJECTIVE, default_weight=1.0
                ),
                "tw_violation": ObjectiveDoc(
                    type=TypeObjective.PENALTY, default_weight=-1000.0
                ),
            },
        )

    def evaluate(self, solution: TSPTWSolution) -> Dict[str, float]:
        """
        Evaluates a solution by calculating the makespan and time window violations.

        This evaluation assumes the distance matrix D[u,v] includes the service time at node u.
        The timeline is calculated as follows:
        1. Arrival at node v = Start of service at u + D[u,v]
        2. Start of service at v = max(Arrival at v, Earliest time for v)
        3. Violation at v = max(0, Start of service at v - Latest time for v)
        """

        # Initialize at depot
        current_node = self.depot_node
        start_service_time = 0.0
        solution.start_service_times = {current_node: 0.0}
        solution.arrival_times = {current_node: 0.0}
        total_violation = 0.0

        # Travel to all customers in the permutation
        for next_node in solution.permutation:
            dist = self.distance_matrix[current_node, next_node]
            arrival_time = start_service_time + dist

            earliest, latest = self.time_windows[next_node]

            start_service_time = max(arrival_time, earliest)

            violation = max(0, start_service_time - latest)
            total_violation += violation

            solution.arrival_times[next_node] = arrival_time
            solution.start_service_times[next_node] = start_service_time

            current_node = next_node

        # Travel back to the depot
        dist_to_depot = self.distance_matrix[current_node, self.depot_node]
        arrival_back_at_depot = start_service_time + dist_to_depot

        earliest_depot, latest_depot = self.time_windows[self.depot_node]

        # Violation for returning to the depot
        depot_return_violation = max(0, arrival_back_at_depot - latest_depot)
        total_violation += depot_return_violation

        solution.makespan = arrival_back_at_depot
        solution.tw_violation = total_violation

        return {"makespan": solution.makespan, "tw_violation": -solution.tw_violation}

    def evaluate_from_encoding(
        self, int_vector: List[int], encoding_name: str
    ) -> Dict[str, float]:
        if encoding_name == "permutation":
            # The encoding gives a permutation of indices from 0 to N-2
            # We map these indices back to the actual customer node IDs
            perm_customers = [self.customers[i] for i in int_vector]
            sol = TSPTWSolution(problem=self, permutation=perm_customers)
        else:
            raise NotImplementedError(f"Encoding '{encoding_name}' is not supported.")

        return self.evaluate(sol)

    def satisfy(self, solution: TSPTWSolution) -> bool:
        if solution.tw_violation is None:
            self.evaluate(solution)
        return solution.tw_violation == 0

    def get_dummy_solution(self) -> TSPTWSolution:
        """Returns a simple, non-random dummy solution (e.g., customers in order)."""
        return TSPTWSolution(problem=self, permutation=self.customers)

    def get_solution_type(self) -> type:
        return TSPTWSolution
