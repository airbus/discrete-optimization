#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import datetime
import json
import os
from typing import Any, Optional

import numpy as np

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.hub_solver.optal.generic_optal import (
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


class OptalTspTwSolver(OptalSolver):
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
        command.append(f"--output-json {self._result_path}")
        return command

    def retrieve_current_solution(self, dict_results: dict) -> TSPTWSolution:
        permutation = dict_results["permutation"]
        return TSPTWSolution(problem=self.problem, permutation=permutation)
