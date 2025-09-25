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
from discrete_optimization.rcpsp_multiskill.problem import (
    MultiskillRcpspProblem,
    MultiskillRcpspSolution,
)

script_ = problem_to_script_path[DoProblemEnum.MSRCPSP]


def dump_to_json(
    problem: MultiskillRcpspProblem,
    one_skill_used_per_worker: bool = False,
    one_worker_per_task: bool = False,
):
    """Exports the problem description to a JSON file."""
    problem_dict = {
        "skills_set": list(problem.skills_set),
        "resources_set": list(problem.resources_set),
        "non_renewable_resources": list(problem.non_renewable_resources),
        "resources_availability": problem.resources_availability,
        "employees": {
            emp_id: emp.to_json() for emp_id, emp in problem.employees.items()
        },
        "mode_details": problem.mode_details,
        "successors": problem.successors,
        "horizon": problem.horizon,
        "tasks_list": problem.tasks_list,
        "employees_list": problem.employees_list,
        "source_task": problem.source_task,
        "sink_task": problem.sink_task,
        "one_skill_used_per_worker": one_skill_used_per_worker,
        "one_worker_per_task": one_worker_per_task,
    }
    return problem_dict


class OptalMSRcpspSolver(OptalSolver):
    problem: MultiskillRcpspProblem

    def __init__(
        self,
        problem: MultiskillRcpspProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self._script_model = script_

    def init_model(self, **args: Any) -> None:
        one_worker_per_task = args.get("one_worker_per_task", False)
        one_skill_used_per_worker = args.get("one_skill_used_per_worker", False)
        output = dump_to_json(
            self.problem,
            one_skill_used_per_worker=one_skill_used_per_worker,
            one_worker_per_task=one_worker_per_task,
        )

        d = datetime.datetime.now().timestamp()
        file_input_path = os.path.join(self.temp_directory, f"tmp-{d}.json")
        logs_path = os.path.join(self.temp_directory, f"tmp-stats-{d}.json")
        result_path = os.path.join(self.temp_directory, f"solution-{d}.json")
        self._logs_path = logs_path
        self._result_path = result_path
        with open(file_input_path, "w") as f:
            json.dump(output, f, indent=4)
        self._file_input = file_input_path
        super().init_model()

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
        employee = dict_results["employeeUsage"]
        employee_skill_usage = dict_results["employeeUsageSkill"]
        schedule = {}
        modes_dict = {}
        tasks = self.problem.get_tasks_list()
        str_to_task = {str(t): t for t in self.problem.tasks_list}
        employee_usage = {}
        for key in start_times:
            schedule[str_to_task[key]] = {
                "start_time": start_times[key],
                "end_time": end_times[key],
            }
            if "modes" in dict_results:
                modes_dict[str_to_task[key]] = int(dict_results["modes"][key])
            else:
                modes_dict[str_to_task[key]] = 1
            if key in employee:
                for emp_key in employee[key]:
                    if emp_key in employee_skill_usage.get(key, {}):
                        non_zeros = employee_skill_usage[key][emp_key]
                    else:
                        non_zeros = self.problem.employees[
                            emp_key
                        ].get_non_zero_skills()
                    usefull_skills = [
                        s
                        for s in self.problem.skills_set
                        if self.problem.mode_details[str_to_task[key]][
                            modes_dict[str_to_task[key]]
                        ].get(s, 0)
                        > 0
                    ]
                    s = [
                        skill
                        for skill in self.problem.skills_set
                        if skill in non_zeros and skill in usefull_skills
                    ]
                    if str_to_task[key] not in employee_usage:
                        employee_usage[str_to_task[key]] = {}
                    employee_usage[str_to_task[key]][emp_key] = set(s)
        schedule[self.problem.source_task] = {"start_time": 0, "end_time": 0}
        max_time = max(end_times.values())
        schedule[self.problem.sink_task] = {
            "start_time": max_time,
            "end_time": max_time,
        }
        return MultiskillRcpspSolution(
            problem=self.problem,
            schedule=schedule,
            modes=modes_dict,
            employee_usage=employee_usage,
        )
