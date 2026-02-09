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


class OptalMSRcpspSolverNode(OptalSolver):
    """Solver for Multi-skill RCPSP using the OptalCP TypeScript API (fallback if Python API is not available)"""
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
        command += ["--output-json", self._result_path]
        return command

    def retrieve_current_solution(self, dict_results: dict) -> Solution:
        start_times = dict_results["startTimes"]
        end_times = dict_results["endTimes"]
        employee = dict_results["employeeUsage"]
        employee_skill_usage = dict_results["employeeUsageSkill"]
        schedule = {}
        modes_dict = {}
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


class OptalMSRcpspSolver(OptalPythonSolver):
    """Solver for MS-RCPSP using the OptalCP Python API (default if OptalCP is installed)"""
    problem: MultiskillRcpspProblem

    def __init__(
        self,
        problem: MultiskillRcpspProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.one_worker_per_task = False
        self.one_skill_used_per_worker = False

    def init_model(self, **args: Any) -> None:
        """Initializes the model and sets flags for worker/skill constraints based on arguments."""
        self.one_worker_per_task = args.get("one_worker_per_task", False)
        self.one_skill_used_per_worker = args.get("one_skill_used_per_worker", False)

    def build_model(self, **kwargs: Any) -> "OptalModel":
        """Builds the OptalCP model for the Multi-skill RCPSP problem."""
        model = cp.Model()
        one_worker_per_task = kwargs.get("one_worker_per_task", self.one_worker_per_task)
        one_skill_used_per_worker = kwargs.get(
            "one_skill_used_per_worker", self.one_skill_used_per_worker
        )
        # Input data preparation
        resource_names = self.problem.resources_set
        non_renewable_set = set(self.problem.non_renewable_resources)
        skill_names = self.problem.skills_set
        employee_names = self.problem.employees_list
        # Placeholder for variables
        self._task_vars = {}
        self._mode_vars = {}
        self._employee_assignment_vars = {}
        self._employee_skill_vars = {}
        ends = []

        renewable_cumuls = {r: [] for r in resource_names if r not in non_renewable_set}
        non_renewable_consumptions = {
            r: [] for r in resource_names if r in non_renewable_set
        }
        employee_cumuls = {emp: [] for emp in employee_names}

        # 1. Tasks, Modes and Employee Assignments
        for taskId in self.problem.tasks_list:
            task_var = model.interval_var(name=f"Task_{taskId}")
            self._task_vars[taskId] = task_var

            task_mode_itvs = []
            self._mode_vars[taskId] = {}
            self._employee_assignment_vars[taskId] = {}
            self._employee_skill_vars[taskId] = {}

            for modeId, modeData in self.problem.mode_details[taskId].items():
                duration = modeData["duration"]
                # Optional interval variable for this specific mode
                mode_var = model.interval_var(
                    name=f"Task_{taskId}_Mode_{modeId}", length=duration, optional=True
                )
                self._mode_vars[taskId][modeId] = mode_var
                task_mode_itvs.append(mode_var)

                # Skill requirements for this mode
                for skill in skill_names:
                    required_level = modeData.get(skill, 0)
                    if required_level > 0:
                        skilled_employees = [
                            emp
                            for emp in employee_names
                            if self.problem.employees[emp].get_skill_level(skill) > 0
                        ]
                        skill_contributions = []
                        for emp in skilled_employees:
                            if emp not in self._employee_assignment_vars[taskId]:
                                # Optional interval variable for assigning this employee to the task
                                emp_task_var = model.interval_var(
                                    name=f"Task_{taskId}_emp_{emp}", optional=True
                                )
                                self._employee_assignment_vars[taskId][
                                    emp
                                ] = emp_task_var
                                # Add to employee cumul for unary resource constraint
                                employee_cumuls[emp].append(emp_task_var.pulse(1))

                                # Sync with task interval
                                # If employee is assigned to the task, 
                                # then the task interval must be present and have the same start/end
                                model.enforce(
                                    model.implies(
                                        model.presence(emp_task_var),
                                        model.start(emp_task_var)
                                        == model.start(task_var),
                                    )
                                )
                                model.enforce(
                                    model.implies(
                                        model.presence(emp_task_var),
                                        model.end(emp_task_var) == model.end(task_var),
                                    )
                                )
                                
                            if one_skill_used_per_worker:
                                # If using one_skill_used_per_worker, 
                                # create separate skill assignment intervals for each employee and skill
                                if emp not in self._employee_skill_vars[taskId]:
                                    self._employee_skill_vars[taskId][emp] = {}
                                # Optional interval variable for this employee using this skill on this task
                                skill_itv = model.interval_var(
                                    name=f"Task_{taskId}_skill_{skill}_emp_{emp}",
                                    optional=True,
                                )
                                # Add to employee skill vars for later constraint enforcement
                                self._employee_skill_vars[taskId][emp][skill] = skill_itv
                                # ensure that if this skill interval is present, then the employee must be assigned to the task
                                skill_contributions.append(model.presence(skill_itv))
                            else:
                                # Otherwise, use the employee assignment variable multiplied by their skill level as contribution.
                                skill_val = self.problem.employees[emp].get_skill_level(skill)
                                skill_contributions.append(model.presence(self._employee_assignment_vars[taskId][emp]) * skill_val)

                        # Skill requirement constraint: 
                        # the sum of contributions from assigned employees must meet the required level
                        model.enforce(
                            model.implies(
                                model.presence(mode_var),
                                model.sum(skill_contributions) >= required_level,
                            )
                        )

                # Renewable/Non-renewable resources
                for resName in resource_names:
                    requirement = modeData.get(resName, 0)
                    if requirement <= 0:
                        continue # No requirement for this resource, skip
                    # If the mode requires this resource, add it to the appropriate cumul or consumption list for constraints
                    if resName in non_renewable_set:
                        non_renewable_consumptions[resName].append(
                            model.presence(mode_var) * requirement
                        )
                    else:
                        renewable_cumuls[resName].append(
                            mode_var.pulse(requirement)
                        )

            # Alternative constraint to link the main task interval with its modes
            model.alternative(task_var, task_mode_itvs)

            # Employee assignment constraints
            if one_skill_used_per_worker:
                for emp, skills_dict in self._employee_skill_vars[taskId].items():
                    skill_itvs = list(skills_dict.values())
                    emp_task_var = self._employee_assignment_vars[taskId][emp]
                    model.enforce(model.sum([model.presence(itv) for itv in skill_itvs]) <= 1)
                    model.alternative(emp_task_var, skill_itvs)
                    pres = model.presence(emp_task_var)
                    for itv in skill_itvs:
                        model.enforce(pres >= model.presence(itv))

            # If using one_worker_per_task, add alternative constraint to ensure at most one employee is assigned to the task
            if one_worker_per_task:
                emp_itvs = list(self._employee_assignment_vars[taskId].values())
                if emp_itvs:
                    model.alternative(task_var, emp_itvs)

            # If this is the sink task, we will set the objective on its end time
            if taskId == self.problem.sink_task:
                ends.append(task_var.end())

        # 2. Precedences
        for taskId, successors in self.problem.successors.items():
            pre_var = self._task_vars[taskId]
            for succId in successors:
                if succId in self._task_vars:
                    model.end_before_start(pre_var, self._task_vars[succId])

        # 3. Resource constraints
        for resName in resource_names:
            availability = self.problem.resources_availability[resName][0]
            if resName in non_renewable_set:
                if non_renewable_consumptions[resName]:
                    model.enforce(
                        model.sum(non_renewable_consumptions[resName]) <= availability
                    )
            else:
                if renewable_cumuls[resName]:
                    model.enforce(
                        model.sum(renewable_cumuls[resName]) <= availability
                    )

        # 4. Employee constraints
        for emp in employee_names:
            if employee_cumuls[emp]:
                model.enforce(model.sum(employee_cumuls[emp]) <= 1)
                # Note: optalcp native also has no_overlap if we have the actual intervals 
                # but sum <= 1 on pulses is equivalent for unary resources.
                itvs = [self._employee_assignment_vars[tid][emp] for tid in self.problem.tasks_list if emp in self._employee_assignment_vars.get(tid, {})]
                if itvs:
                    model.no_overlap(itvs) # Ensure employee cannot be assigned to overlapping tasks (unary resource constraint)

        # 5. Objective: minimize makespan (end time of sink task)
        model.minimize(model.max(ends))
        return model

    def retrieve_current_solution(self, solution: "OptalSolution") -> MultiskillRcpspSolution:
        """Extracts the schedule from the OptalCP solution and constructs a MultiskillRcpspSolution."""
        schedule = {}
        modes_dict = {}
        employee_usage = {}

        for taskId, task_var in self._task_vars.items():
            if taskId != self.problem.source_task and taskId != self.problem.sink_task:
                schedule[taskId] = {
                    "start_time": solution.get_start(task_var),
                    "end_time": solution.get_end(task_var),
                }

            # Mode
            for modeId, mode_var in self._mode_vars[taskId].items():
                if solution.is_present(mode_var):
                    modes_dict[taskId] = int(modeId)
                    break

            # Employees
            assigned_emps = {}
            for emp, emp_task_var in self._employee_assignment_vars[taskId].items():
                if solution.is_present(emp_task_var):
                    skills = set()
                    if taskId in self._employee_skill_vars and emp in self._employee_skill_vars[taskId]:
                        for skill, skill_itv in self._employee_skill_vars[taskId][emp].items():
                            if solution.is_present(skill_itv):
                                skills.add(skill)
                    else:
                        # If not using one_skill_used_per_worker, 
                        # add all skills that were required and that employee has.
                        mode_id = str(modes_dict.get(taskId, 1))
                        mode_data = self.problem.mode_details[taskId][mode_id]
                        for s in self.problem.skills_set:
                            if mode_data.get(s, 0) > 0 and self.problem.employees[emp].get_skill_level(s) > 0:
                                skills.add(s)
                    
                    if skills:
                        assigned_emps[emp] = skills
            
            if assigned_emps:
                employee_usage[taskId] = assigned_emps

        # Add dummy tasks to schedule
        schedule[self.problem.source_task] = {"start_time": 0, "end_time": 0}
        max_end = max([s["end_time"] for s in schedule.values()]) if schedule else 0
        schedule[self.problem.sink_task] = {"start_time": max_end, "end_time": max_end}

        return MultiskillRcpspSolution(
            problem=self.problem,
            schedule=schedule,
            modes=modes_dict,
            employee_usage=employee_usage,
        )
