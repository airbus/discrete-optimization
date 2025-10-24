#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import json
import logging
import os.path
import re
from typing import Any

from discrete_optimization.generic_tools.hub_solver.tempo.tempo_tools import (
    FormatEnum,
    TempoSchedulingSolver,
)
from discrete_optimization.jsp.problem import JobShopProblem, JobShopSolution

logger = logging.getLogger(__name__)


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


def parse_output(solver_output: str, problem: JobShopProblem) -> JobShopSolution:
    matches = re.findall(r"\[(-?\d+),\s*(-?\d+)\]", solver_output)

    if not matches:
        print("Warning: Could not find any schedule information in the solver output.")
        return {}

    # We assume the first value in the pair represents the start time in the
    # solver's relative coordinate system.
    x = [int(match[0]) for match in matches]
    schedule = []
    current_index = 2
    for i in range(problem.n_jobs):
        job_i = problem.list_jobs[i]
        len_sub = len(job_i)
        sched = []
        for j in range(len_sub):
            duration = job_i[j].processing_time
            st = -x[current_index + j]
            sched.append((st, st + duration))

        schedule.append(sched)
        current_index += len_sub
    return JobShopSolution(problem=problem, schedule=schedule)


class TempoJspScheduler(TempoSchedulingSolver):
    problem: JobShopProblem

    def retrieve_solution(
        self, path_to_output: str, process_stdout: str
    ) -> JobShopSolution:
        if os.path.exists(path_to_output):
            dict_ = json.load(open(path_to_output, "r"))
            current_index = 0
            full_sched = []
            for i in range(self.problem.n_jobs):
                job_i = self.problem.list_jobs[i]
                len_sub = len(job_i)
                sched = []
                for j in range(len_sub):
                    duration = job_i[j].processing_time
                    st = dict_["tasks"][current_index + j]["start"][0]
                    sched.append((st, st + duration))
                full_sched.append(sched)
                current_index += len_sub
            return JobShopSolution(problem=self.problem, schedule=full_sched)

    def init_model(self, **kwargs: Any) -> None:
        self._input_format = FormatEnum.JSP
        path = self.get_tmp_folder_path()
        if not os.path.exists(path):
            os.makedirs(path)
        str_repr = from_jsp_to_jsplib(self.problem)
        file_input_path = os.path.join(path, "tmp.txt")
        with open(file_input_path, "w") as f:
            f.write(str_repr)
        self._file_input = file_input_path
