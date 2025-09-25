#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import datetime
import json
import os
from typing import Any, Optional

from discrete_optimization.generic_tools.cp_tools import CpSolver, ParametersCp
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
from discrete_optimization.rcpsp.problem import RcpspProblem, RcpspSolution

script_single_mode = problem_to_script_path[DoProblemEnum.RCPSP]
script_multi_mode = problem_to_script_path[DoProblemEnum.MRCPSP]


def dump_to_json(problem: RcpspProblem):
    mode_details_str_keys = {
        str(task): {str(mode): details for mode, details in modes.items()}
        for task, modes in problem.mode_details.items()
    }
    successors_str_keys = {
        str(task): [str(s) for s in succ] for task, succ in problem.successors.items()
    }

    problem_data = {
        "nbJobs": problem.n_jobs,
        "nbResources": len(problem.resources_list),
        "sourceTask": str(problem.source_task),
        "sinkTask": str(problem.sink_task),
        "tasksList": [str(t) for t in problem.tasks_list],
        "resources": problem.resources,
        "nonRenewableResources": problem.non_renewable_resources,
        "modeDetails": mode_details_str_keys,
        "successors": successors_str_keys,
        "horizon": problem.horizon,
    }
    return problem_data


class OptalRcpspSolver(OptalSolver):
    problem: RcpspProblem

    def __init__(
        self,
        problem: RcpspProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        if self.problem.is_multimode:
            self._script_model = script_multi_mode
        else:
            self._script_model = script_single_mode

    def init_model(self, **args: Any) -> None:
        output = dump_to_json(self.problem)
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
        start_times = dict_results["startTimes"]
        end_times = dict_results["endTimes"]
        rcpsp_schedule = {}
        modes_dict = {}
        tasks = self.problem.get_tasks_list()
        str_to_task = {str(t): t for t in self.problem.tasks_list}
        for key in start_times:
            rcpsp_schedule[str_to_task[key]] = {
                "start_time": start_times[key],
                "end_time": end_times[key],
            }
            if "modes" in dict_results:
                modes_dict[str_to_task[key]] = int(dict_results["modes"][key])
            else:
                modes_dict[str_to_task[key]] = 1
        rcpsp_schedule[self.problem.source_task] = {"start_time": 0, "end_time": 0}
        max_time = max(end_times.values())
        rcpsp_schedule[self.problem.sink_task] = {
            "start_time": max_time,
            "end_time": max_time,
        }
        return RcpspSolution(
            problem=self.problem,
            rcpsp_schedule=rcpsp_schedule,
            rcpsp_modes=[modes_dict[t] for t in self.problem.tasks_list_non_dummy],
        )
