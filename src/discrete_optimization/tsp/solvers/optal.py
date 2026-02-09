#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import datetime
import json
import os
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

try:
    import optalcp as cp
    if TYPE_CHECKING:
        from optalcp import Model as OptalModel, Solution as OptalSolution  # type: ignore
except ImportError:
    cp = None
    
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.hub_solver.optal.generic_optal import (
    OptalPythonSolver,
    OptalSolver
)
from discrete_optimization.generic_tools.hub_solver.optal.model_collections import (
    DoProblemEnum,
    problem_to_script_path,
)
from discrete_optimization.tsp.problem import TspProblem, TspSolution
from discrete_optimization.tsp.utils import build_matrice_distance


def tsp_to_dict(problem: TspProblem, scaling: float = 100):
    """
    Exports the TSP problem to a JSON file, computing and storing the
    full distance matrix.
    """
    # Ensure the distance matrix is computed
    if hasattr(problem, "distance_matrix") and problem.distance_matrix is not None:
        dist_matrix = np.asarray(scaling * problem.distance_matrix, dtype=int)
    else:
        dist_matrix = np.asarray(
            scaling
            * build_matrice_distance(
                problem.node_count, problem.evaluate_function_indexes
            ),
            dtype=int,
        )
    return {
        "node_count": problem.node_count,
        "start_index": problem.start_index,
        "end_index": problem.end_index,
        # Convert numpy array to a standard list of lists for JSON
        "distance_matrix": dist_matrix.tolist(),
    }


class OptalTspSolverNode(OptalSolver):
    """Solver for TSP using the OptalCP TypeScript API (fallback if Python API is not available)"""
    problem: TspProblem

    def __init__(
        self,
        problem: TspProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self._script_model = problem_to_script_path[DoProblemEnum.TSP]

    def init_model(self, scaling: float = 100.0, **args: Any) -> None:
        output = tsp_to_dict(self.problem, scaling=scaling)
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

    def retrieve_current_solution(self, dict_results: dict) -> TspSolution:
        permutation = dict_results["permutation"]
        return TspSolution(
            problem=self.problem,
            start_index=self.problem.start_index,
            end_index=self.problem.end_index,
            permutation=permutation,
        )


class OptalTspSolver(OptalPythonSolver):
    """Solver for TSP using the OptalCP Python API (default if OptalCP is installed)"""
    problem: TspProblem

    def build_model(self, scaling: float = 100.0, **kwargs: Any) -> "OptalModel":
        """Builds the OptalCP model for the TSP problem."""
        model = cp.Model()
        node_count = self.problem.node_count
        start_index = self.problem.start_index
        end_index = self.problem.end_index
        
        # Check if the distance matrix is already computed, otherwise compute it
        if hasattr(self.problem, "distance_matrix") and self.problem.distance_matrix is not None:
            dist_matrix = (scaling * self.problem.distance_matrix).astype(int).tolist()
        else:
            dist_matrix = build_matrice_distance(
                self.problem.node_count, self.problem.evaluate_function_indexes
            )
            dist_matrix = (scaling * np.asarray(dist_matrix)).astype(int).tolist()

        # Create interval variables for each node (except start and end)
        visit_intervals = [
            model.interval_var(length=0, name=f"Visit_{i}") for i in range(node_count)
        ]
        tour_end = model.interval_var(length=0, name="TourEnd")

        # Define the sequence of visits, excluding the start and end nodes
        nodes_in_sequence = [
            i for i in range(node_count) if i != start_index and i != end_index
        ]
        sequence_intervals = [visit_intervals[i] for i in nodes_in_sequence]
        # Create a sequence variable that will enforce the order of visits
        sequence = model.sequence_var(sequence_intervals, nodes_in_sequence)

        # Constraints to ensure the tour starts at start_index and ends at end_index
        model.enforce(visit_intervals[start_index].start() == 0)
        model.no_overlap(sequence, dist_matrix)

        # Add constraints to ensure the correct travel times between nodes based on the distance matrix
        for ni in nodes_in_sequence:
            model.end_before_start(
                visit_intervals[start_index],
                visit_intervals[ni],
                dist_matrix[start_index][ni],
            )
            model.end_before_start(
                visit_intervals[ni], tour_end, dist_matrix[ni][end_index]
            )

        if not nodes_in_sequence and start_index != end_index:
            model.end_before_start(
                visit_intervals[start_index], tour_end, dist_matrix[start_index][end_index]
            )

        # Objective: minimize the total tour length (end time of the tour)
        model.minimize(tour_end.start())
        self._visit_intervals = visit_intervals
        self._nodes_in_sequence = nodes_in_sequence
        return model

    def retrieve_current_solution(self, solution: "OptalSolution") -> TspSolution:
        """Extracts the tour from the OptalCP solution and constructs a TspSolution."""
        sequenced_nodes_with_times = []
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

        return TspSolution(
            problem=self.problem,
            start_index=self.problem.start_index,
            end_index=self.problem.end_index,
            permutation=permutation,
        )
