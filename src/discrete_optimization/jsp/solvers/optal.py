#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  JSP solver using OptalCp solver, see installation instruction on their
import datetime
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
from discrete_optimization.jsp.problem import JobShopProblem, JobShopSolution

script = problem_to_script_path[DoProblemEnum.JSP]


def from_jsp_to_jsplib(problem: JobShopProblem) -> str:
    output = ""

    # --- Header line: number of jobs and machines ---
    output += f"{problem.n_jobs} {problem.n_machines}\n"

    # --- Job lines ---
    for job in problem.list_jobs:
        line_items = []
        for subjob in job:
            line_items.append(str(subjob.machine_id))
            line_items.append(str(subjob.processing_time))
        output += " ".join(line_items) + "\n"

    return output


class OptalJspSolver(OptalSolver):
    problem: JobShopProblem

    def __init__(
        self,
        problem: JobShopProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self._script_model = script

    def retrieve_current_solution(self, dict_results: dict) -> Solution:
        sched = [[] for _ in range(self.problem.n_jobs)]
        for i in range(len(dict_results["startTimes"])):
            for j in range(len(dict_results["startTimes"][i])):
                sched[i].append(
                    (dict_results["startTimes"][i][j], dict_results["endTimes"][i][j])
                )
        sol = JobShopSolution(problem=self.problem, schedule=sched)
        return sol

    def build_command(
        self,
        parameters_cp: Optional[ParametersCp] = None,
        time_limit: int = 10,
        **args: Any,
    ):
        command_list = super().build_command(
            parameters_cp=parameters_cp, time_limit=time_limit, **args
        )
        command_list += ["--outputjsp", self._result_path]
        return command_list

    def init_model(self, **args: Any) -> None:
        output = from_jsp_to_jsplib(self.problem)
        d = datetime.datetime.now().timestamp()
        file_input_path = os.path.join(self.temp_directory, f"tmp-{d}.txt")
        logs_path = os.path.join(self.temp_directory, f"tmp-stats-{d}.json")
        result_path = os.path.join(self.temp_directory, f"solution-{d}.json")
        self._logs_path = logs_path
        self._result_path = result_path
        with open(file_input_path, "w") as f:
            f.write(output)
        self._file_input = file_input_path
