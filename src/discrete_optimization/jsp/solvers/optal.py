#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  JSP solver using OptalCp solver, see installation instruction on their
import datetime
import os
from typing import TYPE_CHECKING, Any, Optional

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
try:
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


class OptalJspSolverNode(OptalSolver):
    """Solver for JSP using the OptalCP TypeScript API (fallback if Python API is not available)"""
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


class OptalJspSolver(OptalPythonSolver):
    """Solver for JSP using the OptalCP Python API (default if OptalCP is installed)"""
    problem: JobShopProblem

    def build_model(self, **kwargs: Any) -> "OptalModel":  
        """Builds the OptalCP model for the JSP problem."""
        model = cp.Model()
        nb_jobs = self.problem.n_jobs
        nb_machines = self.problem.n_machines
        # Placeholders for machine assignments and intervals
        machines = [[] for _ in range(nb_machines)]
        self._all_intervals = [[] for _ in range(nb_jobs)]
        ends = []

        for i, job in enumerate(self.problem.list_jobs):
            prev = None
            for j, subjob in enumerate(job):
                # Create an interval variable for each operation
                operation = model.interval_var(
                    length=subjob.processing_time,
                    name=f"J{i + 1}O{j + 1}M{subjob.machine_id + 1}",
                )
                machines[subjob.machine_id].append(operation)
                self._all_intervals[i].append(operation)
                # Add precedence constraint with the previous operation in the same job
                if prev is not None:
                    model.end_before_start(prev, operation)
                prev = operation
            if prev is not None:
                ends.append(prev.end())
        # Add no-overlap constraints for each machine
        for m in range(nb_machines):
            model.no_overlap(machines[m])
        # Objective: minimize makespan (max of end times of last operations)
        model.minimize(model.max(ends))
        return model

    def retrieve_current_solution(self, solution: "OptalSolution") -> JobShopSolution:
        """Extracts the schedule from the OptalCP solution and constructs a JobShopSolution."""
        sched = [[] for _ in range(self.problem.n_jobs)]
        for i in range(self.problem.n_jobs):
            for j in range(len(self._all_intervals[i])):
                itv = self._all_intervals[i][j]
                sched[i].append((solution.get_start(itv), solution.get_end(itv)))
        return JobShopSolution(problem=self.problem, schedule=sched)
