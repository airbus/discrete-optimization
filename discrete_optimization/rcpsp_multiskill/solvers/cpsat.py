#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import math
from typing import Any, Iterable, List

import numpy as np
from ortools.sat.python.cp_model import (
    Constraint,
    CpModel,
    CpSolverSolutionCallback,
    Domain,
    LinearExpr,
)

from discrete_optimization.generic_tools.do_problem import Problem, Solution
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
)
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp_multiskill.problem import (
    MultiskillRcpspProblem,
    MultiskillRcpspSolution,
    compute_discretize_calendar_skills,
    create_fake_tasks_multiskills,
    discretize_calendar_,
)

logger = logging.getLogger(__name__)


class CpSatMultiskillRcpspSolver(OrtoolsCpSatSolver):
    hyperparameters = [
        CategoricalHyperparameter(
            name="redundant_skill_cumulative", choices=[True, False], default=True
        ),
        CategoricalHyperparameter(
            name="redundant_worker_cumulative", choices=[True, False], default=True
        ),
    ]
    problem: MultiskillRcpspProblem

    def __init__(self, problem: Problem, **kwargs: Any):

        super().__init__(problem, **kwargs)
        self.variables = {}

    def set_lexico_objective(self, obj: str) -> None:
        self.cp_model.Minimize(self.variables[obj])

    def add_lexico_constraint(self, obj: str, value: float) -> Iterable[Constraint]:
        return [self.cp_model.Add(self.variables[obj] <= int(value))]

    @staticmethod
    def implements_lexico_api() -> bool:
        return True

    def get_lexico_objectives_available(self) -> List[str]:
        return ["makespan"]
        # return ["makespan", "cost"]

    def get_lexico_objective_value(self, obj: str, res: ResultStorage) -> float:
        return min(s._internal_obj[obj] for s, _ in res.list_solution_fits)

    def init_model(self, **args: Any) -> None:
        args = self.complete_with_default_hyperparameters(args)
        one_worker_per_task = args.get("one_worker_per_task", False)
        one_skill_per_task = args.get("one_skill_per_task", False)
        redundant_skill_cumulative = args["redundant_skill_cumulative"]
        redundant_worker_cumulative = args["redundant_worker_cumulative"]
        self.cp_model = CpModel()
        self.variables = {}
        self.create_base_variable()
        self.create_opt_variable_modes()
        self.create_employee_intervals(
            one_worker_per_task=one_worker_per_task,
            one_skill_per_task=one_skill_per_task,
        )
        self.create_skills_variables()
        self.fake_tasks, self.fake_tasks_unit = create_fake_tasks_multiskills(
            self.problem
        )
        self.create_constraint_resource()
        if redundant_skill_cumulative:
            self.constraint_redundant_cumulative_skills()
        if redundant_worker_cumulative:
            self.constraint_redundant_cumulative_worker()
        self.create_disjunctive_worker()
        self.create_skills_constraint_to_mode()
        self.create_skills_constraint_worker(**args)
        self.create_skills_constraints_v2(**args)
        self.constraint_precedence()
        self.variables["makespan"] = self.variables["base_variable"]["ends"][
            self.problem.sink_task
        ]
        self.create_workload_variables()
        self.cp_model.Minimize(self.variables["makespan"])

    def create_base_variable(self):
        start_var = {}
        end_var = {}
        duration_var = {}
        interval_var = {}
        for task in self.problem.tasks_list:
            possible_duration = [
                self.problem.mode_details[task][m]["duration"]
                for m in self.problem.mode_details[task]
            ]
            start_var[task] = self.cp_model.NewIntVar(
                lb=0, ub=self.problem.horizon, name=f"start_{task}"
            )
            end_var[task] = self.cp_model.NewIntVar(
                lb=0, ub=self.problem.horizon, name=f"end_{task}"
            )
            duration_var[task] = self.cp_model.NewIntVarFromDomain(
                domain=Domain.FromValues(possible_duration), name=f"duration_{task}"
            )
            interval_var[task] = self.cp_model.NewIntervalVar(
                start=start_var[task],
                size=duration_var[task],
                end=end_var[task],
                name=f"interval_{task}",
            )
        self.variables["base_variable"] = {
            "starts": start_var,
            "ends": end_var,
            "durations": duration_var,
            "intervals": interval_var,
        }

    def create_opt_variable_modes(self):
        if not self.problem.is_multimode:
            self.variables["mode_variable"] = {"is_present": {}, "opt_intervals": {}}
            return
        opt_interval_var = {}
        is_present_var = {}
        for task in self.problem.tasks_list:
            modes = list(self.problem.mode_details[task])
            if len(modes) == 1:
                continue
            is_present_var[task] = {}
            opt_interval_var[task] = {}
            for mode in modes:
                is_present_var[task][mode] = self.cp_model.NewBoolVar(
                    name=f"{task}_{mode}"
                )
                opt_interval_var[task][mode] = self.cp_model.NewOptionalIntervalVar(
                    start=self.variables["base_variable"]["starts"][task],
                    size=self.problem.mode_details[task][mode]["duration"],
                    end=self.variables["base_variable"]["ends"][task],
                    is_present=is_present_var[task][mode],
                    name=f"opt_{task}_{mode}",
                )
        self.variables["mode_variable"] = {
            "is_present": is_present_var,
            "opt_intervals": opt_interval_var,
        }

    def create_employee_intervals(
        self, one_worker_per_task: bool, one_skill_per_task: bool
    ):
        opt_interval_var = {}
        is_present_var = {}
        skills_used_var = {}
        employees_per_skill = {}
        for s in self.problem.skills_set:
            employees_per_skill[s] = {
                e
                for e in self.problem.employees
                if s in self.problem.employees[e].get_non_zero_skills()
            }
        for task in self.problem.tasks_list:
            skills_of_task = set()
            for mode in self.problem.mode_details[task]:
                for skill in self.problem.skills_set:
                    if self.problem.mode_details[task][mode].get(skill, 0) > 0:
                        skills_of_task.add(skill)
            if len(skills_of_task) == 0:
                # no need of employees
                continue
            is_present_var[task] = {}
            opt_interval_var[task] = {}
            skills_used_var[task] = {}
            for worker in self.problem.employees:
                if (
                    one_worker_per_task
                    and any(
                        all(
                            self.problem.employees[worker].get_skill_level(s)
                            >= self.problem.mode_details[task][mode].get(s, 0)
                            for s in skills_of_task
                        )
                        for mode in self.problem.mode_details[task]
                    )
                ) or any(worker in employees_per_skill[s] for s in skills_of_task):
                    skills_used_var[task][worker] = {}
                    is_present_var[task][worker] = self.cp_model.NewBoolVar(
                        name=f"used_{task}_{worker}"
                    )
                    opt_interval_var[task][
                        worker
                    ] = self.cp_model.NewOptionalIntervalVar(
                        start=self.variables["base_variable"]["starts"][task],
                        size=self.variables["base_variable"]["durations"][task],
                        end=self.variables["base_variable"]["ends"][task],
                        is_present=is_present_var[task][worker],
                        name=f"opt_{task}_{worker}",
                    )
                    skills_of_worker = self.problem.employees[
                        worker
                    ].get_non_zero_skills()
                    for s in skills_of_task:
                        if s not in skills_of_worker:
                            # skills_used_var[task][worker][s] = 0
                            continue
                        else:
                            if not one_skill_per_task or len(skills_of_worker) == 1:
                                skills_used_var[task][worker][s] = is_present_var[task][
                                    worker
                                ]
                            else:
                                skills_used_var[task][worker][
                                    s
                                ] = self.cp_model.NewBoolVar(
                                    name=f"skill_{task}_{worker}_{s}"
                                )
                    for s in skills_used_var[task][worker]:
                        self.cp_model.Add(
                            skills_used_var[task][worker][s]
                            <= is_present_var[task][worker]
                        )
                    self.cp_model.AddBoolOr(
                        [
                            skills_used_var[task][worker][s]
                            for s in skills_used_var[task][worker]
                        ]
                    ).OnlyEnforceIf(is_present_var[task][worker])
                    if one_skill_per_task:
                        if len(skills_used_var[task][worker]) >= 1:
                            self.cp_model.AddAtMostOne(
                                [
                                    skills_used_var[task][worker][s]
                                    for s in skills_used_var[task][worker]
                                ]
                            )
            if one_worker_per_task:
                self.cp_model.AddAtMostOne(
                    [is_present_var[task][worker] for worker in is_present_var[task]]
                )
        self.variables["worker_variable"] = {
            "is_present": is_present_var,
            "opt_intervals": opt_interval_var,
            "skills_used": skills_used_var,
        }

    def create_skills_variables(self):
        skills_var = {}
        for task in self.problem.tasks_list:
            skills_of_task = set()
            for mode in self.problem.mode_details[task]:
                for skill in self.problem.skills_set:
                    if self.problem.mode_details[task][mode].get(skill, 0) > 0:
                        skills_of_task.add(skill)
            skills_var[task] = {}
            for s in skills_of_task:
                skills_var[task][s] = self.cp_model.NewIntVar(
                    lb=min(
                        [
                            self.problem.mode_details[task][m].get(s, 0)
                            for m in self.problem.mode_details[task]
                        ]
                    ),
                    ub=max(
                        [
                            self.problem.mode_details[task][m].get(s, 0)
                            for m in self.problem.mode_details[task]
                        ]
                    ),
                    name=f"skills_{task}_{s}",
                )
        self.variables["skills_req"] = skills_var

    def create_workload_variables(self):
        workload = {}
        for emp in self.problem.employees:
            tasks = {
                task
                for task in self.variables["worker_variable"]["is_present"]
                if emp in self.variables["worker_variable"]["is_present"][task]
            }
            workload[emp] = sum(
                [
                    self.problem.mode_details[t][1]["duration"]
                    * self.variables["worker_variable"]["is_present"][t][emp]
                    for t in tasks
                ]
            )
        max_workload = self.cp_model.NewIntVar(
            lb=0, ub=self.problem.horizon, name=f"max_workload"
        )
        min_workload = self.cp_model.NewIntVar(
            lb=0, ub=self.problem.horizon, name=f"min_workload"
        )
        self.cp_model.AddMaxEquality(max_workload, [workload[emp] for emp in workload])
        self.cp_model.AddMinEquality(min_workload, [workload[emp] for emp in workload])
        self.variables["max_workload"] = max_workload
        self.variables["min_workload"] = min_workload

    def create_skills_constraint_to_mode(self):
        for task in self.variables["skills_req"]:
            if task in self.variables["mode_variable"]["is_present"]:
                for mode in self.variables["mode_variable"]["is_present"][task]:
                    for s in self.variables["skills_req"][task]:
                        val = self.problem.mode_details[task][mode].get(s, 0)
                        self.cp_model.Add(
                            self.variables["skills_req"][task][s] == val
                        ).OnlyEnforceIf(
                            self.variables["mode_variable"]["is_present"][task][mode]
                        )

    def create_skills_constraint_worker(self, **args):
        exact_skill = args.get("exact_skill", False)
        slack_skill = args.get("slack_skill", False)
        if slack_skill:
            slack_skill_dict = {}
        for task in self.variables["skills_req"]:
            if slack_skill:
                slack_skill_dict[task] = {}
            for s in self.variables["skills_req"][task]:
                if slack_skill:
                    slack_skill_dict[task][s] = self.cp_model.NewIntVar(
                        lb=0, ub=5, name=f"slack_{task}_{s}"
                    )
                terms = []
                weights = []
                for worker in self.variables["worker_variable"]["is_present"][task]:
                    skill_value = self.problem.employees[worker].get_skill_level(s)
                    if skill_value != 0:
                        terms.append(
                            self.variables["worker_variable"]["is_present"][task][
                                worker
                            ]
                        )
                        weights.append(skill_value)
                if exact_skill:
                    if not slack_skill:
                        self.cp_model.Add(
                            LinearExpr.weighted_sum(terms, weights)
                            == self.variables["skills_req"][task][s]
                        )
                    else:
                        self.cp_model.Add(
                            LinearExpr.weighted_sum(terms, weights)
                            == self.variables["skills_req"][task][s]
                            + slack_skill_dict[task][s]
                        )

                else:
                    if not slack_skill:
                        self.cp_model.Add(
                            LinearExpr.weighted_sum(terms, weights)
                            >= self.variables["skills_req"][task][s]
                        )
                    else:
                        self.cp_model.Add(
                            LinearExpr.weighted_sum(terms, weights)
                            >= self.variables["skills_req"][task][s]
                            + slack_skill_dict[task][s]
                        )
        if slack_skill:
            self.variables["slack_skill_var"] = slack_skill_dict

    def create_skills_constraints_v2(self, **args):
        """
        using skills_used variable
        """
        exact_skill = args.get("exact_skill", False)
        slack_skill = args.get("slack_skill", False)
        if slack_skill:
            slack_skill_dict = {}
        for task in self.variables["skills_req"]:
            if slack_skill:
                slack_skill_dict[task] = {}
            for s in self.variables["skills_req"][task]:
                if slack_skill:
                    slack_skill_dict[task][s] = self.cp_model.NewIntVar(
                        lb=0, ub=5, name=f"slack_{task}_{s}"
                    )
                terms = []
                weights = []
                for worker in self.variables["worker_variable"]["is_present"][task]:
                    if (
                        s
                        in self.variables["worker_variable"]["skills_used"][task][
                            worker
                        ]
                    ):
                        terms.append(
                            self.variables["worker_variable"]["skills_used"][task][
                                worker
                            ][s]
                        )
                        weights.append(
                            self.problem.employees[worker].get_skill_level(s)
                        )
                if exact_skill:
                    if not slack_skill:
                        self.cp_model.Add(
                            LinearExpr.weighted_sum(terms, weights)
                            == self.variables["skills_req"][task][s]
                        )
                    else:
                        self.cp_model.Add(
                            LinearExpr.weighted_sum(terms, weights)
                            == self.variables["skills_req"][task][s]
                            + slack_skill_dict[task][s]
                        )

                else:
                    if not slack_skill:
                        self.cp_model.Add(
                            LinearExpr.weighted_sum(terms, weights)
                            >= self.variables["skills_req"][task][s]
                        )
                    else:
                        self.cp_model.Add(
                            LinearExpr.weighted_sum(terms, weights)
                            >= self.variables["skills_req"][task][s]
                            + slack_skill_dict[task][s]
                        )
        if slack_skill:
            self.variables["slack_skill_var"] = slack_skill_dict

    def constraint_precedence(self):
        for task in self.problem.successors:
            for succ in self.problem.successors[task]:
                self.cp_model.Add(
                    self.variables["base_variable"]["starts"][succ]
                    >= self.variables["base_variable"]["ends"][task]
                )

    def create_constraint_resource(self):
        for r in self.problem.resources_list:
            if r in self.problem.non_renewable_resources:
                self.create_non_renewable_res_constraint(r)
            else:
                self.create_cumulative_resource_constraint(r)

    def create_non_renewable_res_constraint(self, res: str):
        vars_consume = []
        for task in self.problem.tasks_list:
            modes = list(self.problem.mode_details[task].keys())
            if len(modes) == 1:
                if self.problem.mode_details[task][modes[0]].get(res, 0) > 0:
                    vars_consume.append(
                        (1, self.problem.mode_details[task][modes[0]][res])
                    )
            else:
                for mode in modes:
                    if self.problem.mode_details[task][mode].get(res, 0) > 0:
                        vars_consume.append(
                            (
                                self.variables["mode_variable"]["is_present"][task][
                                    mode
                                ],
                                self.problem.mode_details[task][mode][res],
                            )
                        )
        self.cp_model.Add(
            LinearExpr.weighted_sum(
                [x[0] for x in vars_consume], [x[1] for x in vars_consume]
            )
            <= self.problem.get_max_resource_capacity(res)
        )

    def create_cumulative_resource_constraint(self, res: str):
        intervals_consume = []
        for task in self.problem.tasks_list:
            modes = list(self.problem.mode_details[task].keys())
            if len(modes) == 1:
                if self.problem.mode_details[task][modes[0]].get(res, 0) > 0:
                    intervals_consume.append(
                        (
                            self.variables["base_variable"]["intervals"][task],
                            self.problem.mode_details[task][modes[0]][res],
                        )
                    )
            else:
                for mode in modes:
                    if self.problem.mode_details[task][mode].get(res, 0) > 0:
                        intervals_consume.append(
                            (
                                self.variables["mode_variable"]["opt_intervals"][task][
                                    mode
                                ],
                                self.problem.mode_details[task][mode][res],
                            )
                        )
        calendar_tasks = [
            (
                self.cp_model.NewFixedSizeIntervalVar(
                    start=f["start"], size=f["duration"], name="calendar_res"
                ),
                f.get(res, 0),
            )
            for f in self.fake_tasks
            if f.get(res, 0) > 0
        ]
        self.cp_model.AddCumulative(
            [x[0] for x in intervals_consume] + [x[0] for x in calendar_tasks],
            [x[1] for x in intervals_consume] + [x[1] for x in calendar_tasks],
            capacity=self.problem.get_max_resource_capacity(res),
        )

    def create_disjunctive_worker(self):
        for worker in self.problem.employees:
            intervals_consume = []
            for task in self.variables["worker_variable"]["opt_intervals"]:
                if worker in self.variables["worker_variable"]["opt_intervals"][task]:
                    intervals_consume.append(
                        (
                            self.variables["worker_variable"]["opt_intervals"][task][
                                worker
                            ],
                            1,
                        )
                    )
            calendar_tasks = [
                (
                    self.cp_model.NewFixedSizeIntervalVar(
                        start=f["start"], size=f["duration"], name="calendar_res"
                    ),
                    f.get(worker, 0),
                )
                for f in self.fake_tasks_unit
                if f.get(worker, 0) > 0
            ]
            self.cp_model.AddCumulative(
                [x[0] for x in intervals_consume] + [x[0] for x in calendar_tasks],
                [x[1] for x in intervals_consume] + [x[1] for x in calendar_tasks],
                capacity=1,
            )

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
                                self.variables["base_variable"]["intervals"][task],
                                self.problem.mode_details[task][modes[0]][skill],
                            )
                        )
                else:
                    for mode in modes:
                        if self.problem.mode_details[task][mode].get(skill, 0) > 0:
                            intervals_consume.append(
                                (
                                    self.variables["mode_variable"]["opt_intervals"][
                                        task
                                    ][mode],
                                    self.problem.mode_details[task][mode][skill],
                                )
                            )
            calendar_tasks = [
                (
                    self.cp_model.NewFixedSizeIntervalVar(
                        start=f["start"], size=f["duration"], name="calendar_res"
                    ),
                    f.get("value", 0),
                )
                for f in discr_calendar[skill]
                if f.get("value", 0) > 0
            ]
            self.cp_model.AddCumulative(
                [x[0] for x in intervals_consume] + [x[0] for x in calendar_tasks],
                [x[1] for x in intervals_consume] + [x[1] for x in calendar_tasks],
                capacity=int(np.max(dict_calendar_skills[skill])),
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
                            self.variables["base_variable"]["intervals"][task],
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
                                self.variables["mode_variable"]["opt_intervals"][task][
                                    mode
                                ],
                                lb_nb_worker_needed,
                            )
                        )
        calendar_tasks = [
            (
                self.cp_model.NewFixedSizeIntervalVar(
                    start=f["start"], size=f["duration"], name="calendar_res"
                ),
                f.get("value", 0),
            )
            for f in discr_calendar
            if f.get("value", 0) > 0
        ]
        self.cp_model.AddCumulative(
            [x[0] for x in intervals_consume] + [x[0] for x in calendar_tasks],
            [x[1] for x in intervals_consume] + [x[1] for x in calendar_tasks],
            capacity=self.problem.nb_employees,
        )

    def constraint_mode(self):
        for task in self.variables["mode_variable"]["is_present"]:
            self.cp_model.AddExactlyOne(
                [
                    self.variables["mode_variable"]["is_present"][task][m]
                    for m in self.variables["mode_variable"]["is_present"][task]
                ]
            )

    def create_cost_objective_function(self):
        max_salary = max(
            self.problem.employees[x].salary for x in self.problem.employees
        )
        cost_per_tasks = {
            task: self.cp_model.NewIntVar(
                lb=0,
                ub=int(
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
        for task in self.problem.tasks_list:
            modes = list(self.problem.mode_details[task].keys())
            if len(modes) == 1:
                dur = self.problem.mode_details[task][modes[0]]["duration"]
                if task not in self.variables["worker_variable"]["is_present"]:
                    self.cp_model.Add(cost_per_tasks[task] == 0)
                else:
                    workers = [
                        w for w in self.variables["worker_variable"]["is_present"][task]
                    ]
                    self.cp_model.Add(
                        LinearExpr.weighted_sum(
                            [
                                self.variables["worker_variable"]["is_present"][task][w]
                                for w in self.variables["worker_variable"][
                                    "is_present"
                                ][task]
                            ],
                            [
                                dur * int(10 * self.problem.employees[w].salary)
                                for w in workers
                            ],
                        )
                        == cost_per_tasks[task]
                    )
            else:
                workers = [
                    w for w in self.variables["worker_variable"]["is_present"][task]
                ]
                self.cp_model.AddMultiplicationEquality(
                    cost_per_tasks[task],
                    [
                        self.variables["base_variable"]["durations"][task],
                        LinearExpr.weighted_sum(
                            [
                                self.variables["worker_variable"]["is_present"][task][w]
                                for w in self.variables["worker_variable"][
                                    "is_present"
                                ][task]
                            ],
                            [
                                int(10 * self.problem.employees[w].salary)
                                for w in workers
                            ],
                        ),
                    ],
                )
        self.variables["cost"] = sum([cost_per_tasks[t] for t in cost_per_tasks])

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> Solution:
        logger.info(
            f"Current obj {cpsolvercb.ObjectiveValue()}, bound={cpsolvercb.BestObjectiveBound()}"
        )
        modes_dict = {}
        schedule = {}
        employee_usage = {}
        for task in self.variables["base_variable"]["starts"]:
            schedule[task] = {
                "start_time": cpsolvercb.Value(
                    self.variables["base_variable"]["starts"][task]
                ),
                "end_time": cpsolvercb.Value(
                    self.variables["base_variable"]["ends"][task]
                ),
            }
        for task in self.problem.tasks_list:
            modes = list(self.problem.mode_details[task].keys())
            if len(modes) == 1:
                modes_dict[task] = modes[0]
            else:
                for mode in self.variables["mode_variable"]["is_present"][task]:
                    if cpsolvercb.Value(
                        self.variables["mode_variable"]["is_present"][task][mode]
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
            if task in self.variables["worker_variable"]["is_present"]:
                for worker in self.variables["worker_variable"]["is_present"][task]:
                    if cpsolvercb.Value(
                        self.variables["worker_variable"]["is_present"][task][worker]
                    ):
                        sk_nz = self.problem.employees[worker].get_non_zero_skills()
                        if "skills_used" in self.variables["worker_variable"]:
                            contrib = set()
                            for s in self.variables["worker_variable"]["skills_used"][
                                task
                            ][worker]:
                                if cpsolvercb.Value(
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
        for k in self.get_lexico_objectives_available():
            sol._internal_obj[k] = cpsolvercb.Value(self.variables[k])
        return sol
