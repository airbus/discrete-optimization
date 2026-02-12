#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  Optal model for the preemptive rcpsp problem.
from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np

try:
    import optalcp as cp
except ImportError:
    cp = None
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
)
from discrete_optimization.generic_tools.hub_solver.optal.optalcp_tools import (
    OptalCpSolver,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.problem_preemptive import (
    PreemptiveRcpspProblem,
    PreemptiveRcpspSolution,
)
from discrete_optimization.rcpsp.utils import create_fake_tasks


class OptalPreemptiveRcpspSolver(OptalCpSolver):
    problem: PreemptiveRcpspProblem

    def __init__(
        self,
        problem: PreemptiveRcpspProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs,
    ) -> None:
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}
        self.last_sol_do = None

    def implements_lexico_api(self) -> bool:
        return True

    def get_lexico_objective_value(self, obj: str, res: ResultStorage) -> float:
        if obj == "makespan":
            sol: PreemptiveRcpspSolution = res[-1][0]
            return sol.get_max_end_time()
        if obj == "nb_preemption":
            sol: PreemptiveRcpspSolution = res[-1][0]
            return sum(
                [sol.get_number_of_part(task) for task in self.problem.tasks_list]
            )

    def set_lexico_objective(self, obj: str) -> None:
        self.cp_model.minimize(self.variables["objectives"][obj])
        if self.warm_start_solution:
            self.warm_start_solution.set_objective(self.last_sol_do._intern_obj[obj])

    def get_lexico_objectives_available(self) -> list[str]:
        return list(self.variables["objectives"].keys())

    def add_lexico_constraint(self, obj: str, value: float) -> Iterable[Any]:
        self.cp_model.enforce(self.variables["objectives"][obj] <= value)

    def init_model(self, **kwargs: Any) -> None:
        self.cp_model = cp.Model()
        self.create_preempt_variables(
            max_nb_preemption=kwargs.get("max_nb_preemption", None)
        )
        self.constraint_convention_variables()
        self.create_modes_variables()
        self.create_resource_consumption_variables()
        self.create_duration_variables()
        self.constraint_variable_to_duration()
        self.constraint_precedence()
        self.constraint_resource()
        nb_preemption = self.cp_model.sum(
            [
                self.cp_model.presence(self.variables["intervals"][t][i])
                for t in self.variables["intervals"]
                for i in range(len(self.variables["intervals"][t]))
            ]
        )
        self.variables["objectives"] = {
            "makespan": self.cp_model.end(
                self.variables["main_interval"][self.problem.sink_task]
            ),
            "nb_preemption": nb_preemption,
        }
        self.cp_model.minimize(self.variables["objectives"]["makespan"])

    def create_modes_variables(self):
        modes_dict = {}
        for t in self.problem.tasks_list:
            modes_dict[t] = {}
            modes = list(self.problem.mode_details[t].keys())
            nb_modes = len(self.problem.mode_details[t])
            if nb_modes == 1:
                modes_dict[t][modes[0]] = 1
            else:
                for m in modes:
                    modes_dict[t][m] = self.cp_model.bool_var(name=f"mode_{t}_{m}")
                self.cp_model.enforce(
                    self.cp_model.sum([modes_dict[t][m] for m in modes]) == 1
                )
        self.variables["modes"] = modes_dict

    def create_resource_consumption_variables(self):
        modes_var = self.variables["modes"]
        resource_consumption_dict = {}
        for t in self.problem.tasks_list:
            resource_consumption_dict[t] = {}
            modes = list(self.problem.mode_details[t].keys())
            nb_modes = len(self.problem.mode_details[t])
            if nb_modes == 1:
                for r in self.problem.resources_list:
                    cons = self.problem.mode_details[t][modes[0]].get(r, 0)
                    if cons > 0:
                        resource_consumption_dict[t][r] = cons
            else:
                potential_resources = set(
                    [
                        r
                        for r in self.problem.resources_list
                        if any(
                            self.problem.mode_details[t][m].get(r, 0) > 0 for m in modes
                        )
                    ]
                )
                for r in potential_resources:
                    values = [self.problem.mode_details[t][m].get(r, 0) for m in modes]
                    resource_consumption_dict[t][r] = self.cp_model.int_var(
                        min=min(values),
                        max=max(values),
                        name=f"resource_consumption_{t}_{r}",
                    )
                for m in modes_var[t]:
                    for r in potential_resources:
                        cons = self.problem.mode_details[t][m].get(r, 0)
                        self.cp_model.enforce(
                            self.cp_model.implies(
                                modes_var[t][m], resource_consumption_dict[t][r] == cons
                            )
                        )
        self.variables["resource_consumption"] = resource_consumption_dict

    def create_duration_variables(self):
        modes_var = self.variables["modes"]
        duration_dict = {}
        for t in self.problem.tasks_list:
            modes = list(self.problem.mode_details[t].keys())
            nb_modes = len(self.problem.mode_details[t])
            if nb_modes == 1:
                duration_dict[t] = self.problem.mode_details[t][modes[0]]["duration"]
            else:
                potential_durations = list(
                    set([self.problem.mode_details[t][m]["duration"] for m in modes])
                )
                duration_dict[t] = self.cp_model.int_var(
                    min=min(potential_durations),
                    max=max(potential_durations),
                    name=f"duration_{t}",
                )
                for m in modes_var[t]:
                    dur = self.problem.mode_details[t][m]["duration"]
                    self.cp_model.enforce(
                        self.cp_model.implies(modes_var[t][m], duration_dict[t] == dur)
                    )
        self.variables["duration"] = duration_dict

    def create_preempt_variables(self, max_nb_preemption: int | None = None) -> None:
        presences = {}
        intervals = {}
        main_interval = {}
        for t in self.problem.tasks_list:
            possible_durations = [
                self.problem.mode_details[t][m]["duration"]
                for m in self.problem.mode_details[t]
            ]
            max_duration = max(possible_durations)
            if max_nb_preemption is None:
                nb_preemption = max_duration + 1  # Naive
            else:
                nb_preemption = min(max_nb_preemption, max_duration + 1)
            if max_duration == 0:
                min_duration = 0
            else:
                min_duration = 1
            intervals[t] = [
                self.cp_model.interval_var(
                    start=(0, self.problem.horizon),
                    end=(0, self.problem.horizon),
                    length=(min_duration, max_duration),
                    optional=True,
                    name=f"interval_{t}_{i}",
                )
                for i in range(nb_preemption)
            ]
            main_interval[t] = self.cp_model.interval_var(
                start=(0, self.problem.horizon),
                end=(0, self.problem.horizon),
                length=(min(possible_durations), None),
                optional=False,
                name=f"interval_{t}",
            )
            self.cp_model.span(main_interval[t], intervals[t])
            presences[t] = [
                self.cp_model.presence(intervals[t][i]) for i in range(nb_preemption)
            ]
        self.variables["presences"] = presences
        self.variables["intervals"] = intervals
        self.variables["main_interval"] = main_interval

    def constraint_convention_variables(self):
        for t in self.variables["intervals"]:
            nb_preemption = len(self.variables["presences"][t])
            # self.cp_model.enforce(self.cp_model.presence(self.variables["intervals"][t][0]))
            self.cp_model._itv_presence_chain(self.variables["intervals"][t])
            modes = list(self.problem.mode_details[t].keys())
            self.cp_model.no_overlap(self.variables["intervals"][t])
            for i in range(nb_preemption - 1):
                self.cp_model.end_before_start(
                    self.variables["intervals"][t][i],
                    self.variables["intervals"][t][i + 1],
                )
                self.cp_model.enforce(
                    self.cp_model.presence(self.variables["intervals"][t][i])
                    >= self.cp_model.presence(self.variables["intervals"][t][i + 1])
                )
                # self.cp_model.enforce(self.cp_model.implies(self.cp_model.presence(self.variables["intervals"][t][i+1]),
                #                                             self.cp_model.start(self.variables["intervals"][t][i + 1])>=
                #                                             self.cp_model.end(self.variables["intervals"][t][i])+1))

    def constraint_variable_to_duration(self):
        for t in self.variables["presences"]:
            nb_preemption = len(self.variables["presences"][t])
            self.cp_model.enforce(
                self.cp_model.sum(
                    [
                        self.cp_model.guard(
                            self.cp_model.length(self.variables["intervals"][t][i]), 0
                        )
                        for i in range(nb_preemption)
                    ]
                )
                == self.variables["duration"][t]
            )

    def constraint_precedence(self):
        for t in self.problem.successors:
            for succ in self.problem.successors[t]:
                self.cp_model.end_before_start(
                    self.variables["main_interval"][t],
                    self.variables["main_interval"][succ],
                )

    def constraint_resource(self):
        fake_tasks = create_fake_tasks(self.problem)
        for r in self.problem.resources:
            if r not in self.problem.non_renewable_resources:
                self.constraint_resource_cumulative(resource=r, fake_tasks=fake_tasks)
            else:
                self.constraint_resource_non_renewable(resource=r)

    def constraint_resource_cumulative(
        self, resource: str, fake_tasks: list[dict[str, int]]
    ):
        potential_tasks = [
            t
            for t in self.variables["resource_consumption"]
            if resource in self.variables["resource_consumption"][t]
        ]
        intervals = [
            self.cp_model.pulse(
                self.variables["intervals"][t][i],
                self.variables["resource_consumption"][t][resource],
            )
            for t in potential_tasks
            for i in range(len(self.variables["intervals"][t]))
        ]
        fake_tasks_of_interest = [
            self.cp_model.pulse(
                interval=self.cp_model.interval_var(
                    start=f["start"],
                    end=f["start"] + f["duration"],
                    length=f["duration"],
                    optional=False,
                ),
                height=f.get(resource, 0),
            )
            for f in fake_tasks
            if f.get(resource, 0) > 0
        ]
        capa = self.problem.get_max_resource_capacity(resource)
        self.cp_model.enforce(
            self.cp_model.sum(intervals + fake_tasks_of_interest) <= capa
        )

    def constraint_resource_non_renewable(self, resource: str):
        potential_tasks = [
            t
            for t in self.variables["resource_consumption"]
            if resource in self.variables["resource_consumption"][t]
        ]
        capa = self.problem.get_max_resource_capacity(resource)
        self.cp_model.enforce(
            self.cp_model.sum(
                [
                    self.variables["resource_consumption"][t][resource]
                    for t in potential_tasks
                ]
            )
            <= capa
        )

    def retrieve_solution(self, result: cp.SolveResult) -> PreemptiveRcpspSolution:
        # self.warm_start_solution = result.solution
        self.warm_start_solution = result.solution
        modes_dict = {}
        schedule = {}
        for t in self.variables["intervals"]:
            sched = []
            for i in range(len(self.variables["intervals"][t])):
                if result.solution.is_present(self.variables["intervals"][t][i]):
                    sched.append(
                        result.solution.get_value(self.variables["intervals"][t][i])
                    )
            schedule[t] = {
                "starts": [x[0] for x in sched],
                "ends": [x[1] for x in sched],
            }
            for m in self.variables["modes"][t]:
                if isinstance(self.variables["modes"][t][m], int):
                    modes_dict[t] = m
                elif result.solution.get_value(self.variables["modes"][t][m]):
                    modes_dict[t] = m
        modes = [modes_dict[t] for t in self.problem.tasks_list_non_dummy]
        sol = PreemptiveRcpspSolution(
            problem=self.problem, rcpsp_modes=modes, rcpsp_schedule=schedule
        )
        sol._intern_obj = {
            obj: self.get_lexico_objective_value(
                obj, self.create_result_storage([(sol, 0)])
            )
            for obj in self.get_lexico_objectives_available()
        }
        self.last_sol_do = sol
        return sol


def compute_binary_calendar_per_tasks(
    problem: PreemptiveRcpspProblem,
) -> tuple[dict[tuple, np.ndarray], dict[tuple[int, int], tuple]]:
    availability = {
        res: np.array(problem.get_resource_availability_array(res))
        for res in problem.resources_list
    }
    resource_calendar_dict = {
        problem.resources_list[i]: availability[problem.resources_list[i]] > 0
        for i in range(len(problem.resources_list))
    }
    cumulative_calendar_dict = {
        r: np.cumsum(resource_calendar_dict[r]) for r in resource_calendar_dict
    }
    durations = {
        (i, m): None for i in problem.tasks_list for m in problem.mode_details[i]
    }
    task_mode_to_calendar = {}
    for i in problem.tasks_list:
        for m in problem.mode_details[i]:
            duration = problem.mode_details[i][m]["duration"]
            resource_non_zeros = [
                r
                for r in problem.resources_list
                if problem.mode_details[i][m].get(r, 0) > 0
            ]
            if len(resource_non_zeros) == 0:
                durations[i, m] = ([], {duration: [[0, problem.horizon]]})
            elif len(resource_non_zeros) == 1:
                # One resource pool is used.
                res_consumption = problem.mode_details[i][m][resource_non_zeros[0]]
                c = availability[resource_non_zeros[0]] >= res_consumption
                resource_calendar_dict[(resource_non_zeros[0], res_consumption)] = c
                task_mode_to_calendar[i, m] = (resource_non_zeros[0], res_consumption)
            else:
                tuple_res = tuple(
                    [(r, problem.mode_details[i][m][r]) for r in resource_non_zeros]
                )
                if tuple_res not in resource_calendar_dict:
                    # For the first resource in the tuple, b  "availability >= consumption"
                    first_res_id, first_consumption = tuple_res[0]
                    b = availability[first_res_id] >= first_consumption
                    for res_id, cons in tuple_res[1:]:
                        b &= availability[res_id] >= cons
                    resource_calendar_dict[tuple_res] = b
                    cumulative_calendar_dict[tuple_res] = np.cumsum(
                        resource_calendar_dict[tuple_res]
                    )
                task_mode_to_calendar[i, m] = tuple_res
    return resource_calendar_dict, task_mode_to_calendar


class OptalCalendarPreemptiveRcpspSolver(OptalCpSolver):
    problem: PreemptiveRcpspProblem

    def __init__(
        self,
        problem: PreemptiveRcpspProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs,
    ) -> None:
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}
        self.last_sol_do = None
        self.calendar_step_functions = {}
        self.resource_calendar_dict, self.task_mode_to_calendar = (
            compute_binary_calendar_per_tasks(self.problem)
        )

    def create_calendar_step_function(self):
        self.calendar_step_functions = {}
        for i, m in self.task_mode_to_calendar:
            key = self.task_mode_to_calendar[i, m]
            if key in self.calendar_step_functions:
                continue
            array = self.resource_calendar_dict[key]
            initial_value = array[0]
            list_val = [(0, int(initial_value))]
            for t in range(1, array.shape[0]):
                if array[t] != array[t - 1]:
                    list_val.append((t, int(array[t])))
            self.calendar_step_functions[key] = self.cp_model.step_function(list_val)

    def create_main_variables(self):
        intervals = {}
        opt_intervals = {}
        for t in self.problem.tasks_list:
            possible_durations = [
                self.problem.mode_details[t][m]["duration"]
                for m in self.problem.mode_details[t]
            ]
            min_duration = min(possible_durations)
            if min_duration == max(possible_durations) == 0:
                intervals[t] = self.cp_model.interval_var(
                    start=(0, self.problem.horizon),
                    end=(min_duration, self.problem.horizon),
                    length=0,
                    optional=False,
                    name=f"intervals_{t}",
                )
            else:
                intervals[t] = self.cp_model.interval_var(
                    start=(0, self.problem.horizon),
                    end=(min_duration, self.problem.horizon),
                    length=(min_duration, None),
                    optional=False,
                    name=f"intervals_{t}",
                )
            opt_intervals[t] = {}
            modes = list(self.problem.mode_details[t].keys())
            if len(modes) == 1:
                opt_intervals[t][modes[0]] = intervals[t]
            else:
                for m in self.problem.mode_details[t]:
                    dur = self.problem.mode_details[t][m]["duration"]
                    opt_intervals[t][m] = self.cp_model.interval_var(
                        start=(0, self.problem.horizon),
                        end=(min_duration, self.problem.horizon),
                        length=(dur, None),
                        optional=True,
                        name=f"intervals_{t}_{m}",
                    )
                self.cp_model.alternative(
                    intervals[t], [opt_intervals[t][m] for m in opt_intervals[t]]
                )
        self.variables["intervals"] = intervals
        self.variables["opt_intervals"] = opt_intervals

    def create_precedence(self):
        for t in self.problem.successors:
            for succ in self.problem.successors[t]:
                self.cp_model.end_before_start(
                    self.variables["intervals"][t], self.variables["intervals"][succ]
                )

    def constraint_duration_integral(self):
        for t in self.problem.tasks_list:
            for m in self.problem.mode_details[t]:
                dur = self.problem.mode_details[t][m]["duration"]
                if dur > 0:
                    calendar_key = self.task_mode_to_calendar[(t, m)]
                    function = self.calendar_step_functions[calendar_key]
                    self.cp_model.enforce(
                        self.cp_model.guard(
                            self.cp_model.integral(
                                function, self.variables["opt_intervals"][t][m]
                            ),
                            dur,
                        )
                        == dur
                    )

    def init_model(self, **kwargs: Any) -> None:
        self.cp_model = cp.Model()
        self.create_calendar_step_function()
        self.create_main_variables()
        self.create_precedence()
        self.constraint_duration_integral()
        self.constraint_resource()
        self.variables["objectives"] = {
            "makespan": self.cp_model.end(
                self.variables["intervals"][self.problem.sink_task]
            )
        }
        self.cp_model.minimize(self.variables["objectives"]["makespan"])

    def constraint_resource(self):
        fake_tasks = create_fake_tasks(self.problem)
        for r in self.problem.resources:
            if r not in self.problem.non_renewable_resources:
                self.constraint_resource_cumulative(resource=r, fake_tasks=fake_tasks)
            else:
                self.constraint_resource_non_renewable(resource=r)

    def constraint_resource_cumulative(
        self, resource: str, fake_tasks: list[dict[str, int]]
    ):
        max_capacity = self.problem.get_max_resource_capacity(resource)
        potential_tasks = [
            (t, i, self.problem.mode_details[t][i].get(resource, 0))
            for t in self.variables["opt_intervals"]
            for i in self.variables["opt_intervals"][t]
            if self.problem.mode_details[t][i].get(resource, 0) > 0
        ]
        different_calendar_values = set(
            [f.get(resource, 0) for f in fake_tasks if f.get(resource, 0) > 0]
        )
        for diff_value in different_calendar_values:
            calendar_pulse = [
                self.cp_model.pulse(
                    interval=self.cp_model.interval_var(
                        start=f["start"],
                        end=f["start"] + f["duration"],
                        length=f["duration"],
                        optional=False,
                    ),
                    height=f.get(resource, 0),
                )
                for f in fake_tasks
                if 0 < f.get(resource, 0) <= diff_value
            ]
            task_pulse = [
                self.cp_model.pulse(
                    interval=self.variables["opt_intervals"][t][m], height=q
                )
                for t, m, q in potential_tasks
                if q + diff_value <= max_capacity
            ]
            if len(task_pulse) == 0:
                continue
            self.cp_model.enforce(
                self.cp_model.sum(task_pulse + calendar_pulse) <= max_capacity
            )

    def constraint_resource_non_renewable(self, resource: str):
        potential_tasks = [
            (t, m, self.problem.mode_details[t][m].get(resource, 0))
            for t in self.variables["opt_intervals"]
            for m in self.variables["opt_intervals"][t]
            if self.problem.mode_details[t][m].get(resource, 0) > 0
        ]
        capa = self.problem.get_max_resource_capacity(resource)

        self.cp_model.enforce(
            self.cp_model.sum(
                [
                    self.cp_model.presence(self.variables["opt_intervals"][t][m]) * q
                    for t, m, q in potential_tasks
                ]
            )
            <= capa
        )
        # self.cp_model.enforce(self.cp_model.sum([self.cp_model.step_at_start(self.variables["opt_intervals"][t][m],
        #                                                                     q)
        #                                         for t,m,q in potential_tasks]) <= capa)

    def retrieve_solution(self, result: cp.SolveResult) -> PreemptiveRcpspSolution:
        # self.warm_start_solution = result.solution
        self.warm_start_solution = result.solution
        modes_dict = {}
        schedule = {}
        for t in self.variables["intervals"]:
            sched = []
            st, end = result.solution.get_value(self.variables["intervals"][t])
            schedule[t] = {"starts": [st], "ends": [end]}
            modes = list(self.problem.mode_details[t].keys())
            if len(modes) == 1:
                modes_dict[t] = modes[0]
            else:
                for m in self.variables["opt_intervals"][t]:
                    if result.solution.is_present(
                        self.variables["opt_intervals"][t][m]
                    ):
                        modes_dict[t] = m
        modes = [modes_dict[t] for t in self.problem.tasks_list_non_dummy]
        sol = PreemptiveRcpspSolution(
            problem=self.problem, rcpsp_modes=modes, rcpsp_schedule=schedule
        )
        self.last_sol_do = sol
        return sol
