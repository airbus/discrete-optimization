#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import time
from collections.abc import Callable
from itertools import product
from typing import Any, Optional

from ortools.math_opt.python import mathopt

from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.lp_tools import (
    OrtoolsMathOptMilpSolver,
    ParametersMilp,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp_multiskill.problem import (
    MultiskillRcpspProblem,
    MultiskillRcpspSolution,
    tree,
)

logger = logging.getLogger(__name__)


class MathOptMultiskillRcpspSolver(OrtoolsMathOptMilpSolver):
    problem: MultiskillRcpspProblem

    def __init__(
        self,
        problem: MultiskillRcpspProblem,
        params_objective_function: ParamsObjectiveFunction = None,
        **kwargs,
    ):
        super().__init__(
            problem=problem,
            params_objective_function=params_objective_function,
            **kwargs,
        )
        self.variable_decision = {}
        self.constraints_dict = {"lns": []}

    def init_model(self, **args):
        self.model = self.create_empty_model(name="mrcpsp")
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
                mode: self.add_binary_variable(name=f"mode_{task},{mode}")
                for mode in self.problem.mode_details[task]
            }
            for task in self.problem.mode_details
        }

        self.start_times = {
            task: {
                mode: {
                    t: self.add_binary_variable(name=f"start_{task},{mode},{t}")
                    for t in times
                }
                for mode in self.problem.mode_details[task]
            }
            for task in self.problem.mode_details
        }
        # you have to choose one starting date :
        for task in self.start_times:
            self.add_linear_constraint(
                self.construct_linear_sum(
                    self.start_times[task][mode][t]
                    for mode in self.start_times[task]
                    for t in self.start_times[task][mode]
                )
                == 1
            )
            for mode in self.modes[task]:
                self.add_linear_constraint(
                    self.modes[task][mode]
                    == self.construct_linear_sum(
                        self.start_times[task][mode][t]
                        for t in self.start_times[task][mode]
                    )
                )
        self.durations = {
            task: self.add_integer_variable(name="duration_" + str(task))
            for task in self.modes
        }
        self.start_times_task = {
            task: self.add_integer_variable(name=f"start_time_{task}")
            for task in self.start_times
        }
        self.end_times_task = {
            task: self.add_integer_variable(name=f"end_time_{task}")
            for task in self.start_times
        }

        for task in self.start_times:
            self.add_linear_constraint(
                self.construct_linear_sum(
                    self.start_times[task][mode][t] * t
                    for mode in self.start_times[task]
                    for t in self.start_times[task][mode]
                )
                == self.start_times_task[task]
            )
            self.add_linear_constraint(
                self.end_times_task[task]
                - self.start_times_task[task]
                - self.durations[task]
                == 0
            )

        for task in self.durations:
            self.add_linear_constraint(
                self.construct_linear_sum(
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
                            ] = self.add_binary_variable(
                                name=f"employee_{employee}{task}{mode}{t}{s}",
                            )
                            task_in_employee_usage.add(task)
                            self.add_linear_constraint(
                                self.employee_usage[(employee, task, mode, t, s)]
                                - self.modes[task][mode]
                                <= 0
                            )
                            self.add_linear_constraint(
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
                                self.add_linear_constraint(
                                    self.employee_usage[(employee, task, mode, t, s)]
                                    == 0
                                )
        employees = set([x[0] for x in self.employee_usage])

        # can't work on overlapping tasks.
        for emp, t in product(employees, times):
            self.add_linear_constraint(
                self.construct_linear_sum(
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
            self.add_linear_constraint(
                self.construct_linear_sum(
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
            self.add_linear_constraint(
                self.construct_linear_sum(
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
                self.add_linear_constraint(
                    self.construct_linear_sum(
                        self.employee_usage[x]
                        * self.problem.employees[x[0]].dict_skill[s].skill_value
                        for x in employee_usage_keys
                    )
                    >= self.construct_linear_sum(
                        self.modes[task][mode]
                        * self.problem.mode_details[task][mode].get(s, 0)
                        for mode in self.modes[task]
                    )
                )
        for (j, s) in list_edges:
            self.add_linear_constraint(
                self.start_times_task[s] - self.end_times_task[j] >= 0
            )
        self.set_model_objective(
            self.start_times_task[max(self.start_times_task)],
            minimize=True,
        )

    def retrieve_current_solution(
        self,
        get_var_value_for_current_solution: Callable[[Any], float],
        get_obj_value_for_current_solution: Callable[[], float],
    ) -> MultiskillRcpspSolution:
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
        return MultiskillRcpspSolution(
            problem=self.problem,
            modes=modes_task,
            schedule=rcpsp_schedule,
            employee_usage=employee_usage_solution,
        )

    def solve(
        self,
        parameters_milp: Optional[ParametersMilp] = None,
        time_limit: Optional[float] = 30.0,
        **args,
    ) -> ResultStorage:
        if self.model is None:
            logger.info("Init LP model ")
            t = time.time()
            self.init_model(greedy_start=False, **args)
            logger.info(f"LP model initialized...in {time.time() - t} seconds")
        return super().solve(
            parameters_milp=parameters_milp, time_limit=time_limit, **args
        )

    def convert_to_variable_values(
        self, solution: MultiskillRcpspSolution
    ) -> dict[mathopt.Variable, int]:
        hinted_value: dict[mathopt.Variable, int] = {}
        # we define the hint only for task present in solution schedule
        # (potentially only partially defined if infeasible for example)
        for task in solution.schedule:
            mode = solution.modes[task]
            start_time = solution.schedule[task]["start_time"]
            end_time = solution.schedule[task]["end_time"]
            duration = end_time - start_time
            for other_mode, var in self.modes[task].items():
                if other_mode == mode:
                    hinted_value[var] = 1
                else:
                    hinted_value[var] = 0
            for other_mode, vars in self.start_times[task].items():
                if other_mode == mode:
                    for t, var in vars.items():
                        if t == start_time:
                            hinted_value[var] = 1
                        else:
                            hinted_value[var] = 0
                else:
                    for t, var in vars.items():
                        hinted_value[var] = 0
            hinted_value[self.start_times_task[task]] = start_time
            hinted_value[self.end_times_task[task]] = end_time
            hinted_value[self.durations[task]] = duration
        return hinted_value
