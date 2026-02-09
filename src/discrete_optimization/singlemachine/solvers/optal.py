#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import datetime
import json
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


class OptalSingleMachineSolverNode(OptalSolver):
    """Solver for Single Machine Weighted Tardiness using the OptalCP TypeScript API (fallback if Python API is not available)"""
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
        command += ["--output-json", self._result_path]
        return command

        return WTSolution(problem=self.problem, schedule=schedule_list)


class OptalSingleMachineSolver(OptalPythonSolver):
    """Solver for Single Machine Weighted Tardiness using the OptalCP Python API (default if OptalCP is installed)"""
    problem: WeightedTardinessProblem

    def build_model(self, **kwargs: Any) -> "OptalModel":
        """Builds the OptalCP model for the Single Machine Weighted Tardiness problem."""
        model = cp.Model()
        num_jobs = self.problem.num_jobs
        processing_times = self.problem.processing_times
        weights = self.problem.weights
        due_dates = self.problem.due_dates
        release_dates = self.problem.release_dates
        max_time = sum(processing_times) + max(release_dates) + 1000  # Reasonable upper bound

        job_vars = []
        weighted_tardiness_exprs = []

        for i in range(num_jobs):
            # Interal variable for the job
            # release_dates[i] <= start <= end <= max_time
            job_var = model.interval_var(
                length=int(processing_times[i]),
                start=(int(release_dates[i]), int(max_time)),
                name=f"Job_{i}",
                optional=False,
            )
            job_vars.append(job_var)

            # tardiness = max(0, end - due_date)
            tardiness = model.max([0, job_var.end() - due_dates[i]])
            weighted_tardiness_exprs.append(tardiness * weights[i])
        
        # No overlap constraint (single machine)
        model.no_overlap(job_vars)
        
        # Objective: minimize the total weighted tardiness
        model.minimize(model.sum(weighted_tardiness_exprs))

        self._job_vars = job_vars
        return model

    def retrieve_current_solution(self, solution: "OptalSolution") -> WTSolution:
        """Extracts the schedule from the OptalCP solution and constructs a WTSolution."""
        schedule_list = []
        for i in range(self.problem.num_jobs):
            job_var = self._job_vars[i]
            start = solution.get_start(job_var)
            end = solution.get_end(job_var)
            schedule_list.append((start, end))

        return WTSolution(problem=self.problem, schedule=schedule_list)
