#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import datetime
import os
from typing import Any, Optional

from discrete_optimization.fjsp.problem import FJobShopProblem, FJobShopSolution
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
from discrete_optimization.jsp.problem import Subjob

script = problem_to_script_path[DoProblemEnum.FJSP]


def deparse_file(problem: FJobShopProblem, original_header_float: float = 0.0) -> str:
    """
    Writes an FJobShopProblem object to a string in the .fjs format.

    Args:
        problem: The FJobShopProblem object to write.
        original_header_float: Optional float value from the original file's header.

    Returns:
        A string containing the problem data in .fjs format.
    """
    output_lines = []

    # --- Header line ---
    header = f"{problem.n_jobs} {problem.n_machines} {original_header_float}"
    output_lines.append(header)

    # --- Job lines ---
    for job in problem.list_jobs:
        line_items = []
        # Number of sub-jobs (operations) for this job
        line_items.append(str(len(job.sub_jobs)))

        for subjob_options in job.sub_jobs:
            # Number of machine alternatives for this sub-job
            line_items.append(str(len(subjob_options)))

            for option in subjob_options:
                # Machine ID (convert back to 1-indexed) and processing time
                line_items.append(str(option.machine_id + 1))
                line_items.append(str(option.processing_time))

        output_lines.append(" ".join(line_items))

    return "\n".join(output_lines) + "\n"


class OptalFJspSolver(OptalSolver):
    problem: FJobShopProblem

    def __init__(
        self,
        problem: FJobShopProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self._script_model = script

    def init_model(self, **args: Any) -> None:
        output = deparse_file(self.problem)
        d = datetime.datetime.now().timestamp()
        file_input_path = os.path.join(self.temp_directory, f"tmp-{d}.txt")
        logs_path = os.path.join(self.temp_directory, f"tmp-stats-{d}.json")
        result_path = os.path.join(self.temp_directory, f"solution-{d}.json")
        self._logs_path = logs_path
        self._result_path = result_path
        with open(file_input_path, "w") as f:
            f.write(output)
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
        command.append(f"--fjssp-json {self._result_path}")
        return command

    def retrieve_current_solution(self, dict_results: dict) -> Solution:
        start_times = dict_results["startTimes"]
        machine_assignment = dict_results["machineAssignments"]
        index = 0
        full_schedule = []
        for i in range(self.problem.n_jobs):
            sched_i = []
            for j in range(len(self.problem.list_jobs[i].sub_jobs)):
                machine = machine_assignment[index]
                sj: Subjob = next(
                    sub
                    for sub in self.problem.list_jobs[i].sub_jobs[j]
                    if sub.machine_id == machine - 1
                )
                duration = sj.processing_time
                tuple_sched = (
                    start_times[index],
                    start_times[index] + duration,
                    machine - 1,
                )
                index += 1
                sched_i.append(tuple_sched)
            full_schedule.append(sched_i)
        return FJobShopSolution(problem=self.problem, schedule=full_schedule)
