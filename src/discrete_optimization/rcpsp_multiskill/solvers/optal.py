#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import math
from typing import Any, Hashable, Optional

import numpy as np
import optalcp as cp

from discrete_optimization.generic_tasks_tools.solvers.optalcp_tasks_solver import (
    AllocationOptalSolver,
    MultimodeOptalSolver,
    SchedulingOptalSolver,
)
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
)
from discrete_optimization.rcpsp_multiskill.problem import (
    MultiskillRcpspProblem,
    MultiskillRcpspSolution,
    Task,
    UnaryResource,
    compute_discretize_calendar_skills,
    create_fake_tasks_multiskills,
    discretize_calendar_,
)

logger = logging.getLogger(__name__)


class OptalMSRcpspSolver(
    SchedulingOptalSolver[Task],
    MultimodeOptalSolver[Task],
    AllocationOptalSolver[Task, UnaryResource],
):
    hyperparameters = [
        CategoricalHyperparameter(
            name="redundant_skill_cumulative", choices=[True, False], default=True
        ),
        CategoricalHyperparameter(
            name="redundant_worker_cumulative", choices=[True, False], default=True
        ),
    ]
    problem: MultiskillRcpspProblem

    def __init__(
        self,
        problem: MultiskillRcpspProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **args,
    ):
        super().__init__(problem, params_objective_function, **args)
        self.variables = {}
        self.fake_tasks = None  # object to store calendar of resources
        self.fake_tasks_unit = None  # object to store calendar of units

    def init_model(self, **args: Any) -> None:
        super().init_model(**args)
        args = self.complete_with_default_hyperparameters(args)
        one_worker_per_task = args.get("one_worker_per_task", False)
        one_skill_per_task = args.get("one_skill_per_task", False)
        redundant_skill_cumulative = args["redundant_skill_cumulative"]
        redundant_worker_cumulative = args["redundant_worker_cumulative"]
        self.cp_model = cp.Model()
        self.create_main_and_modes_variables()
        self.constraint_precedence()
        self.create_employee_intervals(
            one_worker_per_task=one_worker_per_task,
            one_skill_per_task=one_skill_per_task,
        )
        self.create_skills_req()
        self.create_skills_allocated_constraint()
        self.fake_tasks, self.fake_tasks_unit = create_fake_tasks_multiskills(
            self.problem
        )
        self.add_resource_calendar_constraint()
        self.add_employees_calendar_constraint()
        if redundant_worker_cumulative:
            self.constraint_redundant_cumulative_worker()
        if redundant_skill_cumulative:
            self.constraint_redundant_cumulative_skills()
        self.variables["objs"] = {"makespan": self.get_global_makespan_variable()}
        self.cp_model.minimize(self.variables["objs"]["makespan"])

    def create_main_and_modes_variables(self):
        interval_var = {}
        opt_interval_var = {}
        # TODO : there might be some tighter lb/ub for the tasks.
        horizon = self.problem.get_makespan_upper_bound()
        for task in self.problem.tasks_list:
            modes_list = list(self.problem.mode_details[task].keys())
            potential_duration = set(
                [self.problem.mode_details[task][m]["duration"] for m in modes_list]
            )
            lb_duration = min(potential_duration)
            ub_duration = max(potential_duration)
            length_input = (lb_duration, ub_duration)
            if lb_duration == ub_duration:
                length_input = lb_duration
            interval_var[task] = self.cp_model.interval_var(
                start=(0, horizon),
                end=(0, horizon),
                length=length_input,
                optional=False,
                name=f"interval_{task}",
            )
            opt_interval_var[task] = {}
            if len(modes_list) == 1:
                opt_interval_var[task][modes_list[0]] = interval_var[task]  # Dummy
                continue
            for m in self.problem.mode_details[task]:
                dur = self.problem.mode_details[task][m]["duration"]
                opt_interval_var[task][m] = self.cp_model.interval_var(
                    start=(0, horizon),
                    end=(0, horizon),
                    length=dur,
                    optional=True,
                    name=f"opt_interval_{task}_{m}",
                )
            self.cp_model.alternative(
                interval_var[task],
                [opt_interval_var[task][m] for m in opt_interval_var[task]],
            )
        self.variables["interval_var"] = interval_var
        self.variables["opt_interval_var"] = opt_interval_var

    def constraint_precedence(self):
        for t in self.problem.successors:
            for t_succ in self.problem.successors[t]:
                self.cp_model.end_before_start(
                    self.get_task_interval_variable(t),
                    self.get_task_interval_variable(t_succ),
                )

    def all_skills_for_task(self, task: Task) -> list[Hashable]:
        skills_of_task = set()
        for mode in self.problem.mode_details[task]:
            for skill in self.problem.skills_set:
                if self.problem.mode_details[task][mode].get(skill, 0) > 0:
                    skills_of_task.add(skill)
        return skills_of_task

    def worker_can_do_task(
        self, task: Task, worker: UnaryResource, one_worker_per_task: bool
    ) -> bool:
        skills_of_task = self.all_skills_for_task(task)
        return (
            one_worker_per_task
            and any(
                all(
                    self.problem.employees[worker].get_skill_level(s)
                    >= self.problem.mode_details[task][mode].get(s, 0)
                    for s in skills_of_task
                )
                for mode in self.problem.mode_details[task]
            )
        ) or any(worker in self.problem.employees_per_skill[s] for s in skills_of_task)

    def create_employee_intervals(
        self, one_worker_per_task: bool, one_skill_per_task: bool
    ):
        opt_interval_var = {}
        skills_used_var = {}
        for task in self.problem.tasks_list:
            modes_list = list(self.problem.mode_details[task].keys())
            potential_duration = set(
                [self.problem.mode_details[task][m]["duration"] for m in modes_list]
            )
            lb_duration = min(potential_duration)
            ub_duration = max(potential_duration)
            length_input = (lb_duration, ub_duration)
            if lb_duration == ub_duration:
                length_input = lb_duration
            skills_of_task = self.all_skills_for_task(task)
            if len(skills_of_task) == 0:
                # no need of employees
                continue
            opt_interval_var[task] = {}
            skills_used_var[task] = {}
            for worker in self.problem.employees:
                if self.worker_can_do_task(
                    task, worker, one_worker_per_task=one_worker_per_task
                ):
                    skills_used_var[task][worker] = {}
                    opt_interval_var[task][worker] = self.cp_model.interval_var(
                        start=(0, self.problem.horizon),
                        end=(0, self.problem.horizon),
                        length=length_input,
                        optional=True,
                        name=f"opt_{task}_{worker}",
                    )
                    # Synchro
                    self.cp_model.start_at_start(
                        opt_interval_var[task][worker],
                        self.variables["interval_var"][task],
                    )
                    # self.cp_model.end_at_end(opt_interval_var[task][worker],
                    #                         self.variables["interval_var"][task])
                    skills_of_worker = self.problem.employees[
                        worker
                    ].get_non_zero_skills()
                    for s in skills_of_task:
                        if s not in skills_of_worker:
                            continue
                        else:
                            if not one_skill_per_task or len(skills_of_worker) == 1:
                                # We use all skills.
                                skills_used_var[task][worker][s] = (
                                    self.cp_model.bool_var(
                                        name=f"skill_{task}_{worker}_{s}"
                                    )
                                )
                                self.cp_model.enforce(
                                    skills_used_var[task][worker][s]
                                    == self.cp_model.presence(
                                        opt_interval_var[task][worker]
                                    )
                                )
                            else:
                                skills_used_var[task][worker][s] = (
                                    self.cp_model.bool_var(
                                        name=f"skill_{task}_{worker}_{s}"
                                    )
                                )
                    for s in skills_used_var[task][worker]:
                        self.cp_model.enforce(
                            skills_used_var[task][worker][s]
                            <= self.cp_model.presence(opt_interval_var[task][worker])
                        )

                    self.cp_model.enforce(
                        self.cp_model.presence(opt_interval_var[task][worker])
                        == self.cp_model.max(
                            [
                                skills_used_var[task][worker][s]
                                for s in skills_used_var[task][worker]
                            ]
                        )
                    )
                    if one_skill_per_task:
                        if len(skills_used_var[task][worker]) >= 1:
                            self.cp_model.enforce(
                                self.cp_model.sum(
                                    [
                                        skills_used_var[task][worker][s]
                                        for s in skills_used_var[task][worker]
                                    ]
                                )
                                <= 1
                            )
            if one_worker_per_task:
                self.cp_model.alternative(
                    self.get_task_interval_variable(task),
                    [
                        opt_interval_var[task][worker]
                        for worker in opt_interval_var[task]
                    ],
                )
        self.variables["worker_variable"] = {
            "opt_intervals": opt_interval_var,
            "skills_used": skills_used_var,
        }

    def create_skills_req(self):
        skills_req = {}
        for task in self.problem.tasks_list:
            skills_all = self.all_skills_for_task(task)
            skills_req[task] = {}
            for skill in skills_all:
                levels = [
                    self.problem.mode_details[task][m].get(skill, 0)
                    for m in self.problem.mode_details[task]
                ]
                min_ = min(levels)
                max_ = max(levels)
                skills_req[task][skill] = self.cp_model.int_var(
                    min=min_, max=max_, name=f"skill_{task}_{skill}"
                )
                for mode in self.problem.mode_details[task]:
                    mode_is_present = self.get_task_mode_is_present_variable(task, mode)
                    req_level = self.problem.mode_details[task][mode].get(skill, 0)
                    self.cp_model.enforce(
                        self.cp_model.implies(
                            mode_is_present, skills_req[task][skill] == req_level
                        )
                    )
        self.variables["skills_req"] = skills_req

    def create_skills_allocated_constraint(self):
        skills_req = self.variables["skills_req"]
        skills_used = self.variables["worker_variable"]["skills_used"]
        for task in self.problem.tasks_list:
            for skill in skills_req[task]:
                sum_skills_employee = self.cp_model.sum(
                    [
                        skills_used[task][worker][skill]
                        for worker in skills_used[task]
                        if skill in skills_used[task][worker]
                    ]
                )
                self.cp_model.enforce(sum_skills_employee >= skills_req[task][skill])

    def add_resource_calendar_constraint(self):
        for r in self.problem.resources_list:
            if r in self.problem.non_renewable_resources:
                self.add_non_renewable_resources_constraint(r)
            else:
                self.add_renewable_resources_constraint(r)

    def add_employees_calendar_constraint(self):
        worker_intervals = self.variables["worker_variable"]["opt_intervals"]
        for e in self.problem.employees_list:
            calendar_tasks = [
                self.cp_model.interval_var(
                    start=f["start"],
                    end=f["start"] + f["duration"],
                    length=f["duration"],
                )
                for f in self.fake_tasks_unit
                if f.get(e, 0) > 0
            ]
            tasks = [
                worker_intervals[task][e]
                for task in worker_intervals
                if e in worker_intervals[task]
            ]
            self.cp_model.no_overlap(tasks + calendar_tasks)

    def add_non_renewable_resources_constraint(self, res: str):
        tasks = [
            (
                self.variables["opt_interval"][task][mode],
                self.problem.mode_details[task][mode].get(res, 0),
            )
            for task in self.variables["opt_interval"]
            for mode in self.variables["opt_interval"][task]
            if self.problem.mode_details[task][mode].get(res, 0) > 0
        ]
        cumul = self.cp_model.sum([self.cp_model.presence(x[0]) * x[1] for x in tasks])
        # TODO, try with step function but here it doesnt seem needed...
        self.cp_model.enforce(cumul <= int(self.problem.get_max_resource_capacity(res)))

    def add_renewable_resources_constraint(self, res: str):
        capacity = int(self.problem.get_max_resource_capacity(res))
        calendar_tasks = [
            (
                self.cp_model.interval_var(
                    start=f["start"],
                    end=f["start"] + f["duration"],
                    length=f["duration"],
                ),
                f.get(res, 0),
            )
            for f in self.fake_tasks
            if f.get(res, 0) > 0
        ]
        tasks = [
            (
                self.variables["opt_interval"][task][mode],
                self.problem.mode_details[task][mode].get(res, 0),
            )
            for task in self.variables["opt_interval"]
            for mode in self.variables["opt_interval"][task]
            if self.problem.mode_details[task][mode].get(res, 0) > 0
        ]
        if capacity == 1:
            self.cp_model.no_overlap(
                [x[0] for x in calendar_tasks] + [x[0] for x in tasks]
            )
        else:
            pulses = [self.cp_model.pulse(x[0], x[1]) for x in calendar_tasks + tasks]
            self.cp_model.enforce(self.cp_model.sum(pulses) <= capacity)

    def constraint_redundant_cumulative_skills(self):
        discr_calendar, dict_calendar_skills = compute_discretize_calendar_skills(
            problem=self.problem
        )
        for skill in self.problem.skills_set:
            intervals_consume = []
            for task in self.problem.tasks_list:
                modes = list(self.problem.mode_details[task].keys())
                if len(modes) == 1:
                    if self.problem.mode_details[task][modes[0]].get(skill, 0) > 0:
                        intervals_consume.append(
                            (
                                self.variables["interval_var"][task],
                                self.problem.mode_details[task][modes[0]][skill],
                            )
                        )
                else:
                    for mode in modes:
                        if self.problem.mode_details[task][mode].get(skill, 0) > 0:
                            intervals_consume.append(
                                (
                                    self.variables["opt_interval_var"][task][mode],
                                    self.problem.mode_details[task][mode][skill],
                                )
                            )
            calendar_tasks = [
                (
                    self.cp_model.interval_var(
                        start=f["start"],
                        end=f["start"] + f["duration"],
                        length=f["duration"],
                    ),
                    f.get("value", 0),
                )
                for f in discr_calendar[skill]
                if f.get("value", 0) > 0
            ]
            pulses = [
                self.cp_model.pulse(x[0], x[1])
                for x in intervals_consume + calendar_tasks
            ]
            self.cp_model.enforce(
                self.cp_model.sum(pulses) <= int(np.max(dict_calendar_skills[skill]))
            )

    def constraint_redundant_cumulative_worker(self):
        some_employee = next(emp for emp in self.problem.employees)
        len_calendar = len(self.problem.employees[some_employee].calendar_employee)
        merged_calendar = np.zeros(len_calendar)
        for emp in self.problem.employees:
            merged_calendar += np.array(self.problem.employees[emp].calendar_employee)
        discr_calendar = discretize_calendar_(merged_calendar)
        intervals_consume = []
        max_skill_over_worker = {s: 0 for s in self.problem.skills_set}
        for emp in self.problem.employees:
            for s in self.problem.skills_set:
                max_skill_over_worker[s] = max(
                    max_skill_over_worker[s],
                    self.problem.employees[emp].get_skill_level(s),
                )
        for task in self.problem.tasks_list:
            modes = list(self.problem.mode_details[task].keys())
            if len(modes) == 1:
                skills_needed = {
                    s: self.problem.mode_details[task][modes[0]].get(s, 0)
                    for s in self.problem.skills_set
                    if self.problem.mode_details[task][modes[0]].get(s, 0) > 0
                }
                if len(skills_needed) > 0:
                    lb_nb_worker_needed = max(
                        [
                            int(math.ceil(skills_needed[s] / max_skill_over_worker[s]))
                            for s in skills_needed
                        ]
                    )
                    intervals_consume.append(
                        (
                            self.variables["interval_var"][task],
                            lb_nb_worker_needed,
                        )
                    )
            else:
                for mode in modes:
                    skills_needed = {
                        s: self.problem.mode_details[task][modes[0]].get(s, 0)
                        for s in self.problem.skills_set
                        if self.problem.mode_details[task][modes[0]].get(s, 0) > 0
                    }
                    if len(skills_needed) > 0:
                        lb_nb_worker_needed = max(
                            [
                                int(
                                    math.ceil(
                                        skills_needed[s] / max_skill_over_worker[s]
                                    )
                                )
                                for s in skills_needed
                            ]
                        )
                        intervals_consume.append(
                            (
                                self.variables["opt_intervals"][task][mode],
                                lb_nb_worker_needed,
                            )
                        )
        calendar_tasks = [
            (
                self.cp_model.interval_var(
                    start=f["start"],
                    end=f["start"] + f["duration"],
                    length=f["duration"],
                ),
                f.get("value", 0),
            )
            for f in discr_calendar
            if f.get("value", 0) > 0
        ]
        self.cp_model.enforce(
            self.cp_model.sum(
                [
                    self.cp_model.pulse(x[0], x[1])
                    for x in intervals_consume + calendar_tasks
                ]
            )
            <= self.problem.nb_employees
        )

    def create_cost_objective_function(self):
        max_salary = max(
            self.problem.employees[x].salary for x in self.problem.employees
        )
        cost_per_tasks = {
            task: self.cp_model.int_var(
                min=0,
                max=int(
                    10
                    * max_salary
                    * max(
                        self.problem.mode_details[task][m]["duration"]
                        for m in self.problem.mode_details[task]
                    )
                ),
                name=f"cost_{task}",
            )
            for task in self.problem.tasks_list
        }
        worker_vars = self.variables["worker_variable"]["opt_intervals"]
        for task in self.problem.tasks_list:
            modes = list(self.problem.mode_details[task].keys())
            if len(modes) == 1:
                dur = self.problem.mode_details[task][modes[0]]["duration"]
                if task not in worker_vars:
                    self.cp_model.enforce(cost_per_tasks[task] == 0)
                else:
                    workers = list(worker_vars[task].keys())
                    self.cp_model.enforce(
                        self.cp_model.sum(
                            [
                                worker_vars[task][w]
                                * dur
                                * int(10 * self.problem.employees[w].salary)
                                for w in workers
                            ]
                        )
                        == cost_per_tasks[task]
                    )
            else:
                workers = [
                    w for w in self.variables["worker_variable"]["is_present"][task]
                ]
                dur = self.cp_model.length(self.variables["interval_var"][task])
                self.cp_model.enforce(
                    self.cp_model.sum(
                        [
                            worker_vars[task][w]
                            * dur
                            * int(10 * self.problem.employees[w].salary)
                            for w in workers
                        ]
                    )
                    == cost_per_tasks[task]
                )

        self.variables["cost"] = sum([cost_per_tasks[t] for t in cost_per_tasks])

    def get_task_interval_variable(self, task: Task) -> cp.IntervalVar:
        return self.variables["interval_var"][task]

    def get_task_mode_is_present_variable(self, task: Task, mode: int) -> cp.BoolExpr:
        if mode in self.variables["opt_interval_var"][task]:
            self.cp_model.presence(self.variables["opt_interval_var"][task][mode])
        return False

    def get_task_unary_resource_is_present_variable(
        self, task: Task, unary_resource: UnaryResource
    ) -> cp.BoolExpr:
        if unary_resource in self.variables["worker_variable"]["opt_intervals"][task]:
            return self.variables["worker_variable"]["opt_intervals"][task][
                unary_resource
            ]
        else:
            return 0

    def retrieve_solution(self, result: cp.SolveResult) -> MultiskillRcpspSolution:
        logger.info(f"Current obj {result.solution.get_objective()}")
        modes_dict = {}
        schedule = {}
        employee_usage = {}
        for task in self.variables["interval_var"]:
            itv = self.get_task_interval_variable(task)
            schedule[task] = {
                "start_time": result.solution.get_start(itv),
                "end_time": result.solution.get_end(itv),
            }
        for task in self.problem.tasks_list:
            modes = list(self.problem.mode_details[task].keys())
            if len(modes) == 1:
                modes_dict[task] = modes[0]
            else:
                for mode in self.variables["mode_variable"]["is_present"][task]:
                    if result.solution.is_present(
                        self.variables["opt_interval_var"][task][mode]
                    ):
                        modes_dict[task] = mode
                        break

        for task in self.problem.tasks_list:
            skills_needed = set(
                [
                    s
                    for s in self.problem.skills_set
                    if self.problem.mode_details[task][modes_dict[task]].get(s, 0) > 0
                ]
            )
            employee_usage[task] = {}
            if task in self.variables["worker_variable"]["opt_intervals"]:
                for worker in self.variables["worker_variable"]["opt_intervals"][task]:
                    if result.solution.is_present(
                        self.variables["worker_variable"]["opt_intervals"][task][worker]
                    ):
                        sk_nz = self.problem.employees[worker].get_non_zero_skills()
                        if "skills_used" in self.variables["worker_variable"]:
                            contrib = set()
                            for s in self.variables["worker_variable"]["skills_used"][
                                task
                            ][worker]:
                                if result.solution.get_value(
                                    self.variables["worker_variable"]["skills_used"][
                                        task
                                    ][worker][s]
                                ):
                                    contrib.add(s)
                        else:
                            contrib = set(sk_nz).intersection(skills_needed)
                        if len(contrib) > 0:
                            employee_usage[task][worker] = contrib
        sol = MultiskillRcpspSolution(
            problem=self.problem,
            schedule=schedule,
            modes=modes_dict,
            employee_usage=employee_usage,
        )
        sol._internal_obj = {}
        return sol
