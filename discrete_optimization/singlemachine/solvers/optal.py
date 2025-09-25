#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import datetime
import json
import os
from typing import Any, Optional

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.hub_solver.optal.generic_optal import (
    OptalSolver,
)
from discrete_optimization.generic_tools.hub_solver.optal.model_collections import (
    DoProblemEnum,
    problem_to_script_path,
)
from discrete_optimization.singlemachine.problem import (
    WeightedTardinessProblem,
    WTSolution,
)


def to_dict(problem: WeightedTardinessProblem):
    """Exports the problem description to a JSON file."""
    return {
        "num_jobs": problem.num_jobs,
        "processing_times": problem.processing_times,
        "weights": problem.weights,
        "due_dates": problem.due_dates,
        "release_dates": problem.release_dates,
    }


class OptalSingleMachineSolver(OptalSolver):
    problem: WeightedTardinessProblem

    def __init__(
        self,
        problem: WeightedTardinessProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self._script_model = problem_to_script_path[DoProblemEnum.SINGLEMACHINE]

    def init_model(self, **args: Any) -> None:
        output = to_dict(self.problem)
        d = datetime.datetime.now().timestamp()
        file_input_path = os.path.join(self.temp_directory, f"tmp-{d}.json")
        logs_path = os.path.join(self.temp_directory, f"tmp-stats-{d}.json")
        result_path = os.path.join(self.temp_directory, f"solution-{d}.json")
        self._logs_path = logs_path
        self._result_path = result_path
        with open(file_input_path, "w") as f:
            json.dump(output, f, indent=4)
        self._file_input = file_input_path

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

    def retrieve_current_solution(self, dict_results: dict) -> Solution:
        schedule_dict = dict_results["schedule"]

        # The schedule needs to be a list of tuples, ordered by job index.
        schedule_list = [None] * self.problem.num_jobs
        for job_id_str, times in schedule_dict.items():
            job_id = int(job_id_str)
            schedule_list[job_id] = (times[0], times[1])

        return WTSolution(problem=self.problem, schedule=schedule_list)
