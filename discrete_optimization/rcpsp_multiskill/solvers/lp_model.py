#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import time
from itertools import product
from typing import Any, Callable, Optional

from mip import BINARY, INTEGER, MINIMIZE, Model, xsum

from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.lp_tools import (
    MilpSolverName,
    ParametersMilp,
    PymipMilpSolver,
    map_solver,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import (
    MS_RCPSPModel,
    MS_RCPSPSolution,
    tree,
)

logger = logging.getLogger(__name__)


class LP_Solver_MRSCPSP(PymipMilpSolver):
    problem: MS_RCPSPModel

    def __init__(
        self,
        problem: MS_RCPSPModel,
        lp_solver: MilpSolverName = MilpSolverName.CBC,
        params_objective_function: ParamsObjectiveFunction = None,
        **kwargs,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.lp_solver = lp_solver
        self.variable_decision = {}
        self.constraints_dict = {"lns": []}

    def init_model(self, **args):
        self.model = Model(
            name="mrcpsp", sense=MINIMIZE, solver_name=map_solver[self.lp_solver]
        )
        sorted_tasks = self.problem.tasks_list
        max_time = args.get("max_time", self.problem.horizon)
        max_duration = max_time

        renewable = {
            r: self.problem.resources_availability[r]
            for r in self.problem.resources_availability
            if r not in self.problem.non_renewable_resources
        }
        non_renewable = {
            r: self.problem.resources_availability[r]
            for r in self.problem.non_renewable_resources
        }
        list_edges = []
        for task in sorted_tasks:
            for suc in self.problem.successors[task]:
                list_edges.append([task, suc])
        times = range(max_duration)
        self.modes = {
            task: {
                mode: self.model.add_var(name=f"mode_{task},{mode}", var_type=BINARY)
                for mode in self.problem.mode_details[task]
            }
            for task in self.problem.mode_details
        }

        self.start_times = {
            task: {
                mode: {
                    t: self.model.add_var(
                        name=f"start_{task},{mode},{t}", var_type=BINARY
                    )
                    for t in times
                }
                for mode in self.problem.mode_details[task]
            }
            for task in self.problem.mode_details
        }
        # you have to choose one starting date :
        for task in self.start_times:
            self.model.add_constr(
                xsum(
                    self.start_times[task][mode][t]
                    for mode in self.start_times[task]
                    for t in self.start_times[task][mode]
                )
                == 1
            )
            for mode in self.modes[task]:
                self.model.add_constr(
                    self.modes[task][mode]
                    == xsum(
                        self.start_times[task][mode][t]
                        for t in self.start_times[task][mode]
                    )
                )
        self.durations = {
            task: self.model.add_var(name="duration_" + str(task), var_type=INTEGER)
            for task in self.modes
        }
        self.start_times_task = {
            task: self.model.add_var(name=f"start_time_{task}", var_type=INTEGER)
            for task in self.start_times
        }
        self.end_times_task = {
            task: self.model.add_var(name=f"end_time_{task}", var_type=INTEGER)
            for task in self.start_times
        }

        for task in self.start_times:
            self.model.add_constr(
                xsum(
                    self.start_times[task][mode][t] * t
                    for mode in self.start_times[task]
                    for t in self.start_times[task][mode]
                )
                == self.start_times_task[task]
            )
            self.model.add_constr(
                self.end_times_task[task]
                - self.start_times_task[task]
                - self.durations[task]
                == 0
            )

        for task in self.durations:
            self.model.add_constr(
                xsum(
                    self.problem.mode_details[task][mode]["duration"]
                    * self.modes[task][mode]
                    for mode in self.modes[task]
                )
                == self.durations[task]
            )
        self.employee_usage = tree()
        task_in_employee_usage = set()
        for employee in self.problem.employees:
            skills_employee = [
                skill
                for skill in self.problem.employees[employee].dict_skill.keys()
                if self.problem.employees[employee].dict_skill[skill].skill_value > 0
            ]
            for task in sorted_tasks:
                for mode in self.problem.mode_details[task]:
                    required_skills = [
                        s
                        for s in self.problem.mode_details[task][mode]
                        if s in self.problem.skills_set
                        and self.problem.mode_details[task][mode][s] > 0
                        and s in skills_employee
                    ]
                    if len(required_skills) == 0:
                        # this employee will be useless anyway, pass
                        continue
                    for s in required_skills:
                        for t in range(max_duration):
                            self.employee_usage[
                                (employee, task, mode, t, s)
                            ] = self.model.add_var(
                                name=f"employee_{employee}{task}{mode}{t}{s}",
                                var_type=BINARY,
                            )
                            task_in_employee_usage.add(task)
                            self.model.add_constr(
                                self.employee_usage[(employee, task, mode, t, s)]
                                - self.modes[task][mode]
                                <= 0
                            )
                            self.model.add_constr(
                                self.employee_usage[(employee, task, mode, t, s)]
                                - self.start_times[task][mode][t]
                                <= 0
                            )
                            len_calendar = len(
                                self.problem.employees[employee].calendar_employee
                            )
                            if any(
                                not self.problem.employees[employee].calendar_employee[
                                    tt
                                ]
                                for tt in range(
                                    t,
                                    min(
                                        t
                                        + self.problem.mode_details[task][mode][
                                            "duration"
                                        ],
                                        len_calendar,
                                    ),
                                )
                            ):
                                self.model.add_constr(
                                    self.employee_usage[(employee, task, mode, t, s)]
                                    == 0
                                )
        employees = set([x[0] for x in self.employee_usage])

        # can't work on overlapping tasks.
        for emp, t in product(employees, times):
            self.model.add_constr(
                xsum(
                    self.employee_usage[x]
                    for x in self.employee_usage
                    if x[0] == emp
                    and x[3]
                    <= t
                    < x[3] + int(self.problem.mode_details[x[1]][x[2]]["duration"])
                )
                <= 1
            )
        # ressource usage limit
        for (r, t) in product(renewable, times):
            self.model.add_constr(
                xsum(
                    int(self.problem.mode_details[task][mode][r])
                    * self.start_times[task][mode][time]
                    for task in self.start_times
                    for mode in self.start_times[task]
                    for time in self.start_times[task][mode]
                    if time
                    <= t
                    < time + int(self.problem.mode_details[task][mode]["duration"])
                )
                <= renewable[r][t]
            )
        # for non renewable ones.
        for r in non_renewable:
            self.model.add_constr(
                xsum(
                    int(self.problem.mode_details[task][mode][r])
                    * self.start_times[task][mode][time]
                    for task in self.start_times
                    for mode in self.start_times[task]
                    for time in self.start_times[task][mode]
                )
                <= non_renewable[r][0]
            )
        for task in self.start_times_task:
            required_skills = [
                (s, mode, self.problem.mode_details[task][mode][s])
                for mode in self.problem.mode_details[task]
                for s in self.problem.mode_details[task][mode]
                if s in self.problem.skills_set
                and self.problem.mode_details[task][mode][s] > 0
            ]
            skills = set([s[0] for s in required_skills])
            for s in skills:
                employee_usage_keys = [
                    v for v in self.employee_usage if v[1] == task and v[4] == s
                ]
                self.model.add_constr(
                    xsum(
                        self.employee_usage[x]
                        * self.problem.employees[x[0]].dict_skill[s].skill_value
                        for x in employee_usage_keys
                    )
                    >= xsum(
                        self.modes[task][mode]
                        * self.problem.mode_details[task][mode].get(s, 0)
                        for mode in self.modes[task]
                    )
                )
        for (j, s) in list_edges:
            self.model.add_constr(
                self.start_times_task[s] - self.end_times_task[j] >= 0
            )
        self.model.objective = self.start_times_task[max(self.start_times_task)]

    def retrieve_current_solution(
        self,
        get_var_value_for_current_solution: Callable[[Any], float],
        get_obj_value_for_current_solution: Callable[[], float],
    ) -> MS_RCPSPSolution:
        rcpsp_schedule = {}
        modes = {}
        results = {}
        employee_usage = {}
        employee_usage_solution = {}
        for task in self.start_times:
            for mode in self.start_times[task]:
                for t, start_time in self.start_times[task][mode].items():
                    value = get_var_value_for_current_solution(start_time)
                    results[(task, mode, t)] = value
                    if value >= 0.5:
                        rcpsp_schedule[task] = {
                            "start_time": int(t),
                            "end_time": int(
                                t + self.problem.mode_details[task][mode]["duration"]
                            ),
                        }
                        modes[task] = mode
        for t in self.employee_usage:
            employee_usage[t] = get_var_value_for_current_solution(
                self.employee_usage[t]
            )
            if employee_usage[t] >= 0.5:
                if t[1] not in employee_usage_solution:
                    employee_usage_solution[t[1]] = {}
                if t[0] not in employee_usage_solution[t[1]]:
                    employee_usage_solution[t[1]][t[0]] = set()
                employee_usage_solution[t[1]][t[0]].add(t[4])
                # (employee, task, mode, time, skill)

        modes = {}
        modes_task = {}
        for t in self.modes:
            for m, mode in self.modes[t].items():
                modes[(t, m)] = get_var_value_for_current_solution(mode)
                if modes[(t, m)] >= 0.5:
                    modes_task[t] = m
        durations = {}
        for t in self.durations:
            durations[t] = get_var_value_for_current_solution(self.durations[t])
        start_time = {}
        for t in self.start_times_task:
            start_time[t] = get_var_value_for_current_solution(self.start_times_task[t])
        end_time = {}
        for t in self.start_times_task:
            end_time[t] = get_var_value_for_current_solution(self.end_times_task[t])
        logger.debug(f"Size schedule : {len(rcpsp_schedule.keys())}")
        logger.debug(
            (
                "results",
                "(task, mode, time)",
                {x: results[x] for x in results if results[x] == 1.0},
            )
        )
        logger.debug(
            (
                "Employee usage : ",
                "(employee, task, mode, time, skill)",
                {
                    x: employee_usage[x]
                    for x in employee_usage
                    if employee_usage[x] == 1.0
                },
            )
        )
        logger.debug(
            (
                "task mode : ",
                "(task, mode)",
                {t: modes[t] for t in modes if modes[t] == 1.0},
            )
        )
        logger.debug(f"durations : {durations}")
        logger.debug(f"Start time {start_time}")
        logger.debug(f"End time {end_time}")
        return MS_RCPSPSolution(
            problem=self.problem,
            modes=modes_task,
            schedule=rcpsp_schedule,
            employee_usage=employee_usage_solution,
        )

    def solve(
        self, parameters_milp: Optional[ParametersMilp] = None, **args
    ) -> ResultStorage:
        if self.model is None:
            logger.info("Init LP model ")
            t = time.time()
            self.init_model(greedy_start=False, **args)
            logger.info(f"LP model initialized...in {time.time() - t} seconds")
        return super().solve(parameters_milp=parameters_milp, **args)
