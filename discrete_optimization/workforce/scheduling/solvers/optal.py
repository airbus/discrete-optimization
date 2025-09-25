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
from discrete_optimization.workforce.scheduling.problem import (
    AllocSchedulingProblem,
    AllocSchedulingSolution,
    export_scheduling_problem_json,
)


def alloc_scheduling_to_dict(problem: AllocSchedulingProblem):
    return export_scheduling_problem_json(problem=problem)


class OptalAllocSchedulingSolver(OptalSolver):
    problem: AllocSchedulingProblem

    def __init__(
        self,
        problem: AllocSchedulingProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self._script_model = problem_to_script_path[DoProblemEnum.WORKFORCE_SCHEDULING]
        self.model_dispersion: bool = False
        self.run_lexico: bool = False

    def init_model(
        self, model_dispersion: bool = False, run_lexico: bool = False, **args: Any
    ) -> None:
        output = alloc_scheduling_to_dict(self.problem)
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
        self.model_dispersion = model_dispersion
        self.run_lexico = run_lexico

    def build_command(
        self,
        parameters_cp: Optional[ParametersCp] = None,
        time_limit: int = 10,
        **args: Any,
    ):
        if "model_dispersion" in args:
            args.pop("model_dispersion")
        if "run_lexico" in args:
            args.pop("run_lexico")
        command = super().build_command(
            parameters_cp=parameters_cp, time_limit=time_limit, **args
        )
        if self.model_dispersion:
            command.append("--model-dispersion")
        if self.run_lexico:
            command.append("--run-lexico")
        command.append(f"--output-json {self._result_path}")
        return command

    def retrieve_current_solution(self, dict_results: dict) -> AllocSchedulingSolution:
        starts = dict_results["startTimes"]
        ends = dict_results["endTimes"]
        allocation = dict_results["teamAssignments"]
        return AllocSchedulingSolution(
            problem=self.problem,
            schedule=np.array([[s, e] for s, e in zip(starts, ends)], dtype=int),
            allocation=np.array([self.problem.teams_to_index[t] for t in allocation]),
        )
