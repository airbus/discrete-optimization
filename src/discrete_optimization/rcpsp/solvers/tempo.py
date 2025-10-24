#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import json
import os
import re
from typing import Any

from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.hub_solver.tempo.tempo_tools import (
    FormatEnum,
    TempoSchedulingSolver,
)
from discrete_optimization.rcpsp.problem import RcpspProblem, RcpspSolution
from discrete_optimization.rcpsp.solvers.rcpsp_solver import RcpspSolver


def deparse_psplib(problem: RcpspProblem, **kwargs) -> str:
    """
    Writes a RcpspProblem object to a string in the .sm file format.

    Args:
        problem: The RcpspProblem object to write.
        **kwargs: Optional keyword arguments for header information like
                  'basedata_filename', 'random_seed', 'duedate', etc.

    Returns:
        A string containing the problem data in .sm format.
    """
    # --- Header ---
    output = (
        "************************************************************************\n"
    )
    output += f"file with basedata            : {kwargs.get('basedata_filename', 'generated.bas')}\n"
    output += (
        f"initial value random generator: {kwargs.get('random_seed', 123456789)}\n"
    )
    output += (
        "************************************************************************\n"
    )

    # --- General Info ---
    n_jobs = problem.n_jobs
    n_renewable = len(problem.resources) - len(problem.non_renewable_resources)
    n_non_renewable = len(problem.non_renewable_resources)

    output += f"projects                      :  1\n"
    output += f"jobs (incl. supersource/sink ):  {n_jobs}\n"
    output += f"horizon                       :  {problem.horizon}\n"
    output += "RESOURCES\n"
    output += f"  - renewable                 :  {n_renewable:<3} R\n"
    output += f"  - nonrenewable              :  {n_non_renewable:<3} N\n"
    output += f"  - doubly constrained        :  0   D\n"
    output += (
        "************************************************************************\n"
    )

    # --- Project Information ---
    # This section has some info not directly in RcpspProblem, so we use defaults.
    output += "PROJECT INFORMATION:\n"
    output += "pronr.  #jobs rel.date duedate tardcost  MPM-Time\n"
    output += f"    1     {n_jobs - 2:<5}      0   {kwargs.get('duedate', problem.horizon):<7}   {kwargs.get('tardcost', 20):<8}  {kwargs.get('mpm_time', problem.horizon)}\n"
    output += (
        "************************************************************************\n"
    )

    # --- Precedence Relations ---
    output += "PRECEDENCE RELATIONS:\n"
    output += "jobnr.    #modes  #successors   successors\n"
    sorted_tasks = sorted(problem.successors.keys())
    for job_id in sorted_tasks:
        successors = problem.successors.get(job_id, [])
        n_successors = len(successors)
        n_modes = len(problem.mode_details.get(job_id, {}))
        successors_str = "  ".join(map(str, successors))
        output += f"  {job_id:<8}  {n_modes:<6}  {n_successors:<11} {successors_str}\n"
    output += (
        "************************************************************************\n"
    )

    # --- Requests/Durations ---
    output += "REQUESTS/DURATIONS:\n"
    resource_names = sorted(problem.resources.keys())
    header_res = "  ".join(resource_names)
    output += f"jobnr. mode duration  {header_res}\n"
    output += (
        "------------------------------------------------------------------------\n"
    )
    sorted_tasks_details = sorted(problem.mode_details.keys())
    for job_id in sorted_tasks_details:
        for mode_id, details in sorted(problem.mode_details[job_id].items()):
            duration = details["duration"]
            res_usage = [str(details.get(res, 0)) for res in resource_names]
            res_usage_str = "    ".join(f"{usage:<2}" for usage in res_usage)
            output += (
                f"  {job_id:<5}  {mode_id:<4}   {duration:<5}     {res_usage_str}\n"
            )
    output += (
        "************************************************************************\n"
    )

    # --- Resource Availabilities ---
    output += "RESOURCEAVAILABILITIES:\n"
    res_avail_header = "  ".join(resource_names)
    output += f"  {res_avail_header}\n"
    availabilities = [str(problem.resources[res]) for res in resource_names]
    availabilities_str = "   ".join(availabilities)
    output += f"   {availabilities_str}\n"
    output += (
        "************************************************************************\n"
    )

    return output


def parse_solver_output(
    solver_output: str, makespan: int, tasks_list_in_output_order: list
) -> dict[int, int]:
    """
    Parses the solver output string to extract a schedule.

    It assumes the solver provides start times in a relative coordinate system
    that needs to be scaled to the actual makespan. It uses linear scaling
    based on the first and last task's values to map to a [0, makespan] schedule.

    Args:
        solver_output: The string output from the solver.
        makespan: The makespan of the project.

    Returns:
        A dictionary mapping each task ID to its calculated start time.
    """
    # Use regex to find all occurrences of [num, num]
    matches = re.findall(r"\[(-?\d+),\s*(-?\d+)\]", solver_output)

    if not matches:
        print("Warning: Could not find any schedule information in the solver output.")
        return {}

    # We assume the first value in the pair represents the start time in the
    # solver's relative coordinate system.
    x = [int(match[0]) for match in matches]

    if len(x) < 2:
        print("Warning: Not enough tasks in solver output to determine schedule.")
        return {}

    x_first = x[0]
    x_last = x[1]

    # The solver's time range for the project.
    solver_range = x_last - x_first

    if solver_range == 0:
        print(
            "Warning: Solver output indicates zero duration between first and last task. Cannot scale schedule."
        )
        # Return a schedule where all tasks start at 0, as there's no range to scale.
        return {i + 1: 0 for i in range(len(x))}

    schedule = {}
    for i, x_i in enumerate(x):
        # Linearly scale the solver's start time to the [0, makespan] range.
        # Formula: s_i = s_start + (s_end - s_start) * (x_i - x_start) / (x_end - x_start)
        # Here, s_start=0, s_end=makespan.
        scaled_time = makespan * (x_i - x_first) / solver_range
        # Start times are typically integers in these problems.
        start_time = round(scaled_time)
        # Task IDs are 1-indexed.
        task_id = i
        schedule[tasks_list_in_output_order[i]] = start_time

    return schedule


class TempoRcpspSolver(TempoSchedulingSolver, RcpspSolver):
    def init_model(self, **kwargs: Any) -> None:
        self._input_format = FormatEnum.PSPLIB
        temp = self.get_tmp_folder_path()
        if not os.path.exists(temp):
            os.makedirs(temp)
        sm_format_str = deparse_psplib(problem=self.problem)
        input_file = os.path.join(temp, "input_tmp.txt")
        with open(input_file, "w") as f:
            f.write(sm_format_str)
        self._file_input = input_file

    def retrieve_solution(self, path_to_output: str, process_stdout: str) -> Solution:
        if os.path.exists(path_to_output):
            dict_ = json.load(open(path_to_output, "r"))
            rcpsp_schedule = {
                self.problem.source_task: {"start_time": 0, "end_time": 0},
                self.problem.sink_task: {
                    "start_time": dict_["tasks"][0]["end"][0],
                    "end_time": dict_["tasks"][0]["end"][0],
                },
            }
            for i in range(1, len(dict_["tasks"])):
                rcpsp_schedule[self.problem.tasks_list[i]] = {
                    "start_time": dict_["tasks"][i]["start"][0],
                    "end_time": dict_["tasks"][i]["end"][0],
                }
            return RcpspSolution(
                problem=self.problem,
                rcpsp_schedule=rcpsp_schedule,
                rcpsp_modes=[1] * self.problem.n_jobs_non_dummy,
            )
