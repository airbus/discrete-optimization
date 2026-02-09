#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import datetime
import json
import os
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
try:
    import optalcp as cp
    import optalcp as cp
    if TYPE_CHECKING:
        from optalcp import Model as OptalModel, Solution as OptalSolution  # type: ignore
except ImportError:
    cp = None
from discrete_optimization.generic_tools.hub_solver.optal.generic_optal import (
    OptalPythonSolver,
    OptalSolver,
)
from discrete_optimization.generic_tools.hub_solver.optal.model_collections import (
    DoProblemEnum,
    problem_to_script_path,
)
from discrete_optimization.tsptw.problem import TSPTWProblem, TSPTWSolution


def tsptw_to_dict(problem: TSPTWProblem, scaling: float = 100):
    """
    Exports the TSP problem to a JSON file, computing and storing the
    full distance matrix.
    """
    # Ensure the distance matrix is computed
    dist_matrix = np.asarray(scaling * problem.distance_matrix, dtype=int)
    return {
        "node_count": problem.nb_nodes,
        "depot": problem.depot_node,
        "time_windows": [
            [int(scaling * x[0]), int(scaling * x[1])] for x in problem.time_windows
        ],
        "distance_matrix": dist_matrix.tolist(),
    }


class OptalTspTwSolverNode(OptalSolver):
    """Solver for TSPTW using the OptalCP TypeScript API (fallback if Python API is not available)"""
    problem: TSPTWProblem

    def __init__(
        self,
        problem: TSPTWProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self._script_model = problem_to_script_path[DoProblemEnum.TSPTW]

    def init_model(self, scaling: float = 100.0, **args: Any) -> None:
        output = tsptw_to_dict(self.problem, scaling=scaling)
        d = datetime.datetime.now().timestamp()
        file_input_path = os.path.join(self.temp_directory, f"tmp-{d}.json")
        logs_path = os.path.join(self.temp_directory, f"tmp-stats-{d}.json")
        result_path = os.path.join(self.temp_directory, f"solution-{d}.json")
        self._logs_path = logs_path
        self._result_path = result_path
        with open(file_input_path, "w") as f:
            json.dump(output, f, indent=4)
        self._file_input = file_input_path
        super().init_model(**args)

    def build_command(
        self,
        parameters_cp: Optional[ParametersCp] = None,
        time_limit: int = 10,
        **args: Any,
    ):
        command = super().build_command(
            parameters_cp=parameters_cp, time_limit=time_limit, **args
        )
        command += ["--output-json", self._result_path]
        return command

    def retrieve_current_solution(self, dict_results: dict) -> TSPTWSolution:
        permutation = dict_results["permutation"]
        return TSPTWSolution(problem=self.problem, permutation=permutation)


class OptalTspTwSolver(OptalPythonSolver):
    """Solver for TSPTW using the OptalCP Python API (if available)"""
    problem: TSPTWProblem

    def build_model(self, scaling: float = 100.0, **kwargs: Any) -> "OptalModel":
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
            model.end_before_start(visit_intervals[ni], tour_end, dist_matrix[ni][depot])

        # Objective: minimize the total tour length (end time of the tour)
        model.minimize(tour_end.start())
        self._visit_intervals = visit_intervals
        self._nodes_in_sequence = nodes_in_sequence
        return model

    def retrieve_current_solution(self, solution: "OptalSolution") -> TSPTWSolution:
        """Extracts the tour from the OptalCP solution and constructs a TSPTWSolution."""
        sequenced_nodes_with_times = []
        # Extract the start times of the visit intervals for the nodes in the sequence
        for node_index in self._nodes_in_sequence:
            sequenced_nodes_with_times.append(
                {
                    "index": node_index,
                    "startTime": solution.get_start(self._visit_intervals[node_index]),
                }
            )
        # Sort the nodes by their start times to get the correct tour order
        sequenced_nodes_with_times.sort(key=lambda x: x["startTime"])
        permutation = [item["index"] for item in sequenced_nodes_with_times]

        return TSPTWSolution(problem=self.problem, permutation=permutation)
