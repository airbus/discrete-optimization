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


class OptalRcpspSolverNode(OptalSolver):
    """Solver for RCPSP using the OptalCP TypeScript API (fallback if Python API is not available)"""
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
        command += ["--output-json", self._result_path]
        return command

    def retrieve_current_solution(self, dict_results: dict) -> Solution:
        start_times = dict_results["startTimes"]
        end_times = dict_results["endTimes"]
        rcpsp_schedule = {}
        modes_dict = {}
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


class OptalRcpspSolver(OptalPythonSolver):
    """Solver for RCPSP using the OptalCP Python API (if available)"""
    problem: RcpspProblem

    def build_model(self, **kwargs: Any) -> "OptalModel":
        """Builds the OptalCP model for the RCPSP problem."""
        model = cp.Model()
        resource_names = self.problem.resources_list
        non_renewable_set = set(self.problem.non_renewable_resources)

        self._jobs_vars = {}
        self._modes_vars = {}
        ends = []
        cumuls = [[] for _ in resource_names]
        total_consumptions = [[] for _ in resource_names]

        # 1. Create variables
        for taskId in self.problem.tasks_list:
            if (
                taskId == self.problem.source_task
                or taskId == self.problem.sink_task
            ):
                continue
            
            itv = model.interval_var(name=f"T{taskId}")
            self._jobs_vars[taskId] = itv
            
            mode_intervals = []
            self._modes_vars[taskId] = {}
            
            for mode, modeData in self.problem.mode_details[taskId].items():
                duration = modeData["duration"]
                itv_mode = model.interval_var(
                    name=f"T{taskId}M{mode}", length=duration, optional=True
                )
                self._modes_vars[taskId][mode] = itv_mode
                mode_intervals.append(itv_mode)
                
                # Resources
                for rIndex, resName in enumerate(resource_names):
                    requirement = modeData.get(resName, 0)
                    if requirement > 0:
                        if resName in non_renewable_set:
                            total_consumptions[rIndex].append(
                                model.presence(itv_mode) * requirement
                            )
                        else:
                            cumuls[rIndex].append(itv_mode.pulse(requirement))
            
            # Add alternative constraint to link the main interval with its modes
            model.alternative(itv, mode_intervals)

        # 2. Precedences
        for taskId in self.problem.tasks_list:
            if (
                taskId == self.problem.source_task
                or taskId == self.problem.sink_task
            ):
                continue
            
            predecessor_itv = self._jobs_vars[taskId]
            successors = self.problem.successors.get(taskId, [])
            is_last = True
            for successorId in successors:
                if (
                    successorId != self.problem.sink_task
                    and successorId in self._jobs_vars
                ):
                    successor_itv = self._jobs_vars[successorId]
                    model.end_before_start(predecessor_itv, successor_itv)
                    is_last = False
            
            if is_last:
                ends.append(predecessor_itv.end())

        # 3. Capacity constraints
        for rIndex, resName in enumerate(resource_names):
            capacity = self.problem.resources[resName]
            if resName in non_renewable_set:
                if total_consumptions[rIndex]:
                    model.enforce(model.sum(total_consumptions[rIndex]) <= capacity)
            else:
                if cumuls[rIndex]:
                    model.enforce(model.sum(cumuls[rIndex]) <= capacity)

        # 4. Objective
        model.minimize(model.max(ends))
        return model

    def retrieve_current_solution(self, solution: "OptalSolution") -> RcpspSolution:
        """Extracts the schedule from the OptalCP solution and constructs an RcpspSolution."""
        start_times = {}
        end_times = {}
        modes_dict = {}
        for taskId, jobVar in self._jobs_vars.items():
            start_times[taskId] = solution.get_start(jobVar)
            end_times[taskId] = solution.get_end(jobVar)
            for mode, modeVar in self._modes_vars[taskId].items():
                if solution.is_present(modeVar):
                    modes_dict[taskId] = int(mode)
                    break
        
        # Add dummies
        start_times[self.problem.source_task] = 0
        end_times[self.problem.source_task] = 0
        
        max_end = max(end_times.values()) if end_times else 0
        start_times[self.problem.sink_task] = max_end
        end_times[self.problem.sink_task] = max_end
        
        rcpsp_schedule = {
            t: {"start_time": start_times[t], "end_time": end_times[t]}
            for t in self.problem.tasks_list
        }
        
        return RcpspSolution(
            problem=self.problem,
            rcpsp_schedule=rcpsp_schedule,
            rcpsp_modes=[modes_dict[t] for t in self.problem.tasks_list_non_dummy],
        )
