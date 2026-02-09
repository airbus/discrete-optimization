#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any

import numpy as np

from discrete_optimization.generic_tools.do_problem import (
    Solution,
)

try:
    import optalcp as cp
except ImportError:
    cp = None
from discrete_optimization.generic_tools.hub_solver.optal.optalcp_tools import (
    OptalCpSolver,
)
from discrete_optimization.tsptw.problem import TSPTWProblem, TSPTWSolution


class OptalTspTwSolver(OptalCpSolver):
    """Solver for TSPTW using the OptalCP Python API (if available)"""

    problem: TSPTWProblem

    def init_model(self, scaling: float = 100.0, **kwargs: Any):
        """Builds the OptalCP model for the TSPTW problem."""
        model = cp.Model()
        node_count = self.problem.nb_nodes
        depot = self.problem.depot_node

        # Scale the distance matrix and time windows for integer representation
        dist_matrix = (scaling * self.problem.distance_matrix).astype(int).tolist()
        time_windows = (scaling * np.asarray(self.problem.time_windows)).astype(int)

        # Create interval variables for each node, with time window constraints
        visit_intervals = []
        for i in range(node_count):
            itv = model.interval_var(length=0, name=f"Visit_{i}")
            model.enforce(itv.start() >= int(time_windows[i][0]))
            model.enforce(itv.start() <= int(time_windows[i][1]))
            visit_intervals.append(itv)

        # Interval variable for the end of the tour
        tour_end = model.interval_var(length=0, name="TourEnd")

        # Constraint: Ensure the tour starts at the depot within its time window
        model.enforce(tour_end.start() >= int(time_windows[depot][0]))
        model.enforce(tour_end.start() <= int(time_windows[depot][1]))

        # Create a sequence variable that will enforce the order of visits
        nodes_in_sequence = [i for i in range(node_count) if i != depot]
        sequence_intervals = [visit_intervals[i] for i in nodes_in_sequence]
        sequence = model.sequence_var(sequence_intervals, nodes_in_sequence)

        # Constraint: enforce the tour starts at the depot and ends at the depot, considering the distance matrix
        model.enforce(visit_intervals[depot].start() == 0)
        model.no_overlap(sequence, dist_matrix)

        for ni in nodes_in_sequence:
            model.end_before_start(
                visit_intervals[depot], visit_intervals[ni], dist_matrix[depot][ni]
            )
            model.end_before_start(
                visit_intervals[ni], tour_end, dist_matrix[ni][depot]
            )

        # Objective: minimize the total tour length (end time of the tour)
        model.minimize(tour_end.start())
        self._visit_intervals = visit_intervals
        self._nodes_in_sequence = nodes_in_sequence
        self.cp_model = model

    def retrieve_solution(self, result: cp.SolveResult) -> Solution:
        """Extracts the tour from the OptalCP solution and constructs a TSPTWSolution."""
        sequenced_nodes_with_times = []
        # Extract the start times of the visit intervals for the nodes in the sequence
        for node_index in self._nodes_in_sequence:
            sequenced_nodes_with_times.append(
                {
                    "index": node_index,
                    "startTime": result.solution.get_start(
                        self._visit_intervals[node_index]
                    ),
                }
            )
        # Sort the nodes by their start times to get the correct tour order
        sequenced_nodes_with_times.sort(key=lambda x: x["startTime"])
        permutation = [item["index"] for item in sequenced_nodes_with_times]

        return TSPTWSolution(problem=self.problem, permutation=permutation)
