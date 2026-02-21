#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  Cp-sat model for the preemptive rcpsp problem.
from typing import Any, Iterable

import numpy as np
from ortools.sat.python.cp_model import CpSolverSolutionCallback, Domain

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.problem_preemptive import (
    PreemptiveRcpspProblem,
    PreemptiveRcpspSolution,
)
from discrete_optimization.rcpsp.utils import create_fake_tasks


class CpSatPreemptiveRcpspSolver(OrtoolsCpSatSolver):
    problem: PreemptiveRcpspProblem

    def __init__(
        self,
        problem: PreemptiveRcpspProblem,
        params_objective_function: ParamsObjectiveFunction | None = None,
        **kwargs,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}

    def init_model(self, **kwargs: Any) -> None:
        super().init_model(**kwargs)
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
        self.variables["objectives"] = {
            "makespan": self.variables["ends"][self.problem.sink_task][0],
            "nb_preemption": sum(
                [
                    self.variables["presences"][t][i]
                    for t in self.variables["presences"]
                    for i in range(len(self.variables["presences"][t]))
                ]
            ),
        }
        self.cp_model.minimize(self.variables["ends"][self.problem.sink_task][0])

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

    def get_lexico_objectives_available(self) -> list[str]:
        return list(self.variables["objectives"].keys())

    def add_lexico_constraint(self, obj: str, value: float) -> Iterable[Any]:
        self.cp_model.add(self.variables["objectives"][obj] <= value)

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
                    modes_dict[t][m] = self.cp_model.NewBoolVar(name=f"mode_{t}_{m}")
                self.cp_model.add_exactly_one([modes_dict[t][m] for m in modes])
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
                    resource_consumption_dict[t][r] = self.cp_model.NewIntVar(
                        lb=min(values),
                        ub=max(values),
                        name=f"resource_consumption_{t}_{r}",
                    )
                for m in modes_var[t]:
                    for r in potential_resources:
                        cons = self.problem.mode_details[t][m].get(r, 0)
                        self.cp_model.add(
                            resource_consumption_dict[t][r] == cons
                        ).only_enforce_if(modes_var[t][m])
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
                duration_dict[t] = self.cp_model.NewIntVarFromDomain(
                    domain=Domain.FromValues(potential_durations), name=f"duration_{t}"
                )
                for m in modes_var[t]:
                    dur = self.problem.mode_details[t][m]["duration"]
                    self.cp_model.add(duration_dict[t] == dur).only_enforce_if(
                        modes_var[t][m]
                    )
        self.variables["duration"] = duration_dict

    def create_preempt_variables(self, max_nb_preemption: int | None = None) -> None:
        starts = {}
        durations = {}
        ends = {}
        presences = {}
        intervals = {}
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
            starts[t] = [
                self.cp_model.NewIntVar(
                    lb=0, ub=self.problem.horizon, name=f"start_{t}_{i}"
                )
                for i in range(nb_preemption)
            ]
            ends[t] = [
                self.cp_model.NewIntVar(
                    lb=0, ub=self.problem.horizon, name=f"end_{t}_{i}"
                )
                for i in range(nb_preemption)
            ]
            # min_duration_preempt = 1
            # if max_duration == 0:
            min_duration_preempt = 0
            durations[t] = [
                self.cp_model.NewIntVar(
                    lb=min_duration_preempt, ub=max_duration, name=f"duration_{t}_{i}"
                )
                for i in range(nb_preemption)
            ]
            presences[t] = [
                self.cp_model.NewBoolVar(name=f"presence_{t}_{i}")
                for i in range(nb_preemption)
            ]
            intervals[t] = [
                self.cp_model.NewOptionalIntervalVar(
                    start=starts[t][i],
                    end=ends[t][i],
                    size=durations[t][i],
                    is_present=presences[t][i],
                    name=f"interval_{t}_{i}",
                )
                for i in range(nb_preemption)
            ]
        self.variables["starts"] = starts
        self.variables["durations"] = durations
        self.variables["ends"] = ends
        self.variables["presences"] = presences
        self.variables["intervals"] = intervals

    def constraint_convention_variables(self):
        for t in self.variables["presences"]:
            nb_preemption = len(self.variables["presences"][t])
            self.cp_model.add(self.variables["presences"][t][0] == 1)
            modes = list(self.problem.mode_details[t].keys())
            potential_durations = list(
                set([self.problem.mode_details[t][m]["duration"] for m in modes])
            )
            if min(potential_durations) > 0:
                self.cp_model.add(self.variables["durations"][t][0] >= 1)
            for i in range(nb_preemption - 1):
                # Ordered intervals and present until some point, then all absent.
                self.cp_model.add(
                    self.variables["presences"][t][i]
                    >= self.variables["presences"][t][i + 1]
                )
                self.cp_model.add(
                    self.variables["ends"][t][i] <= self.variables["starts"][t][i + 1]
                )
                self.cp_model.add(
                    self.variables["ends"][t][i] <= self.variables["ends"][t][i + 1]
                )
                (
                    self.cp_model.add(
                        self.variables["ends"][t][i]
                        < self.variables["starts"][t][i + 1]
                    ).only_enforce_if(self.variables["presences"][t][i + 1])
                )

            for i in range(1, nb_preemption):
                self.cp_model.add(
                    self.variables["durations"][t][i] >= 1
                ).only_enforce_if(self.variables["presences"][t][i])
                self.cp_model.add(
                    self.variables["durations"][t][i] == 0
                ).only_enforce_if(self.variables["presences"][t][i].Not())
                self.cp_model.add(
                    self.variables["starts"][t][i] == self.variables["ends"][t][i - 1]
                ).only_enforce_if(self.variables["presences"][t][i].Not())
                self.cp_model.add(
                    self.variables["ends"][t][i] == self.variables["ends"][t][i - 1]
                ).only_enforce_if(self.variables["presences"][t][i].Not())

    def constraint_variable_to_duration(self):
        for t in self.variables["presences"]:
            nb_preemption = len(self.variables["presences"][t])
            self.cp_model.add(
                sum(self.variables["durations"][t][i] for i in range(nb_preemption))
                == self.variables["duration"][t]
            )

    def constraint_precedence(self):
        for t in self.problem.successors:
            for succ in self.problem.successors[t]:
                self.cp_model.add(
                    self.variables["starts"][succ][0] >= self.variables["ends"][t][-1]
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
            (
                self.variables["intervals"][t][i],
                self.variables["resource_consumption"][t][resource],
            )
            for t in potential_tasks
            for i in range(len(self.variables["intervals"][t]))
        ]
        fake_tasks_of_interest = [
            (
                self.cp_model.NewFixedSizeIntervalVar(
                    start=f["start"], size=f["duration"], name=f"res_"
                ),
                f.get(resource, 0),
            )
            for f in fake_tasks
            if f.get(resource, 0) > 0
        ]
        capa = self.problem.get_max_resource_capacity(resource)
        self.cp_model.add_cumulative(
            [x[0] for x in intervals + fake_tasks_of_interest],
            [x[1] for x in intervals + fake_tasks_of_interest],
            capa,
        )

    def constraint_resource_non_renewable(self, resource: str):
        potential_tasks = [
            t
            for t in self.variables["resource_consumption"]
            if resource in self.variables["resource_consumption"][t]
        ]
        capa = self.problem.get_max_resource_capacity(resource)
        self.cp_model.add(
            sum(
                [
                    self.variables["resource_consumption"][t][resource]
                    for t in potential_tasks
                ]
            )
            <= capa
        )

    def retrieve_solution(
        self, cpsolvercb: CpSolverSolutionCallback
    ) -> PreemptiveRcpspSolution:
        modes_dict = {}
        schedule = {}
        for t in self.variables["starts"]:
            sched = []
            for i in range(len(self.variables["starts"][t])):
                present = cpsolvercb.value(self.variables["presences"][t][i])
                if present:
                    sched.append(
                        (
                            cpsolvercb.value(self.variables["starts"][t][i]),
                            cpsolvercb.value(self.variables["ends"][t][i]),
                        )
                    )
                else:
                    break
            schedule[t] = {
                "starts": [x[0] for x in sched],
                "ends": [x[1] for x in sched],
            }
            modes = list(self.variables["modes"][t].keys())
            if len(modes) == 1:
                modes_dict[t] = modes[0]
            else:
                for m in self.variables["modes"][t]:
                    if cpsolvercb.value(self.variables["modes"][t][m]):
                        modes_dict[t] = m
        modes = [
            modes_dict[t]
            for t in self.problem.tasks_list
            if t not in {self.problem.source_task, self.problem.sink_task}
        ]
        return PreemptiveRcpspSolution(
            problem=self.problem, rcpsp_schedule=schedule, rcpsp_modes=modes
        )


class CpSatCalendarPreemptiveSolver(OrtoolsCpSatSolver):
    problem: PreemptiveRcpspProblem

    def __init__(
        self,
        problem: PreemptiveRcpspProblem,
        params_objective_function: ParamsObjectiveFunction | None = None,
        **kwargs,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}
        self.durations, _, _ = compute_binary_calendar_per_tasks(self.problem)

    def init_model(self, **kwargs: Any) -> None:
        super().init_model(**kwargs)
        self.create_main_variables()
        self.constraint_duration_of_tasks()
        self.constraint_resource()
        self.constraint_precedence()
        self.cp_model.minimize(self.variables["ends"][self.problem.sink_task])

    def create_main_variables(self):
        starts = {}
        ends = {}
        durations = {}
        intervals = {}
        opt_intervals = {}
        opt_durations = {}
        presences = {}
        for t in self.problem.tasks_list:
            starts[t] = self.cp_model.NewIntVar(
                lb=0, ub=self.problem.horizon, name=f"start_{t}"
            )

            ends[t] = self.cp_model.NewIntVar(
                lb=0, ub=self.problem.horizon, name=f"end_{t}"
            )
            positive_durations = sorted(
                list(
                    set(
                        [
                            int(d)
                            for m in self.problem.mode_details[t]
                            for d in self.durations[(t, m)][1]
                            if d >= 0
                        ]
                    )
                )
            )
            durations[t] = self.cp_model.NewIntVarFromDomain(
                domain=Domain.FromValues(positive_durations), name=f"duration_{t}"
            )
            intervals[t] = self.cp_model.NewIntervalVar(
                start=starts[t], end=ends[t], size=durations[t], name=f"interval_{t}"
            )
            modes = list(self.problem.mode_details[t].keys())
            opt_intervals[t] = {}
            opt_durations[t] = {}
            presences[t] = {}
            if len(modes) == 1:
                opt_intervals[t][modes[0]] = intervals[t]
                presences[t][modes[0]] = 1
                opt_durations[t][modes[0]] = durations[t]
            else:
                for m in modes:
                    presences[t][m] = self.cp_model.NewBoolVar(name=f"presence_{t}_{m}")
                    opt_intervals[t][m] = self.cp_model.NewOptionalIntervalVar(
                        start=starts[t],
                        end=ends[t],
                        size=durations[t],
                        is_present=presences[t][m],
                        name=f"opt_interval_{t}_{m}",
                    )
                    self.cp_model.add(
                        durations[t] == opt_durations[t][m]
                    ).only_enforce_if(presences[t][m])
                self.cp_model.add_exactly_one([presences[t][m] for m in presences[t]])

        self.variables["starts"] = starts
        self.variables["ends"] = ends
        self.variables["durations"] = durations
        self.variables["intervals"] = intervals
        self.variables["opt_intervals"] = opt_intervals
        self.variables["opt_durations"] = opt_durations
        self.variables["presences"] = presences

    def constraint_duration_of_tasks(self):
        """
        Tricky constraint : should take into account the partial preemption possibility,
        which makes duration variable based on calendars
        """
        durs = self.durations
        dictionary_indicators = {}
        for task_index, mode in durs:
            d = self.constraint_duration_of_task(
                task_index=task_index,
                mode=mode,
                duration_per_interval=durs[(task_index, mode)][1],
            )
            dictionary_indicators.update(d)
        self.variables["dictionary_indicators"] = dictionary_indicators
        for index in self.variables["presences"]:
            all_key = [
                x for x in self.variables["dictionary_indicators"] if x[0][0] == index
            ]
            self.cp_model.AddExactlyOne(
                [self.variables["dictionary_indicators"][x] for x in all_key]
            )

    def constraint_duration_of_task(
        self,
        task_index: int,
        mode: int,
        duration_per_interval: dict[int, list[tuple[int, int]]],
    ):
        dictionary_indicators = {}
        positive_durations = [d for d in duration_per_interval if d >= 0]
        if len(positive_durations) == 1:
            dur = int(positive_durations[0])
            interval = Domain.FromIntervals(duration_per_interval[dur])
            self.cp_model.AddLinearExpressionInDomain(
                self.variables["starts"][task_index], interval
            ).only_enforce_if(self.variables["presences"][task_index][mode])
            (
                self.cp_model.Add(
                    self.variables["durations"][task_index] == dur
                ).only_enforce_if(self.variables["presences"][task_index][mode])
            )
            dictionary_indicators[((task_index, mode), dur)] = self.variables[
                "presences"
            ][task_index][mode]
        else:
            for possible_duration in duration_per_interval:
                if possible_duration < 0:
                    continue
                interval = Domain.FromIntervals(
                    duration_per_interval[possible_duration]
                )
                dictionary_indicators[((task_index, mode), possible_duration)] = (
                    self.cp_model.NewBoolVar(
                        f"d_{(task_index, mode), possible_duration}"
                    )
                )
                self.cp_model.AddLinearExpressionInDomain(
                    self.variables["starts"][task_index], interval
                ).OnlyEnforceIf(
                    dictionary_indicators[((task_index, mode), possible_duration)]
                )
                self.cp_model.Add(
                    self.variables["durations"][task_index] == int(possible_duration)
                ).OnlyEnforceIf(
                    dictionary_indicators[((task_index, mode), possible_duration)]
                )
            # corrected version (to be confirmed)
            self.cp_model.Add(
                sum([dictionary_indicators[k] for k in dictionary_indicators])
                == self.variables["presences"][task_index][mode]
            )
        return dictionary_indicators

    def constraint_precedence(self):
        for t in self.problem.successors:
            for succ in self.problem.successors[t]:
                self.cp_model.add(
                    self.variables["starts"][succ] >= self.variables["ends"][t]
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
                (
                    self.cp_model.new_fixed_size_interval_var(
                        start=f["start"], size=f["duration"], name=f"dummy"
                    ),
                    f.get(resource, 0),
                )
                for f in fake_tasks
                if 0 < f.get(resource, 0) <= diff_value
            ]
            task_pulse = [
                (self.variables["opt_intervals"][t][m], q)
                for t, m, q in potential_tasks
                if q + diff_value <= max_capacity
            ]
            if len(task_pulse) == 0:
                continue
            self.cp_model.add_cumulative(
                [x[0] for x in task_pulse + calendar_pulse],
                [x[1] for x in task_pulse + calendar_pulse],
                max_capacity,
            )

    def constraint_resource_non_renewable(self, resource: str):
        potential_tasks = [
            (t, m, self.problem.mode_details[t][m].get(resource, 0))
            for t in self.variables["opt_intervals"]
            for m in self.variables["opt_intervals"][t]
            if self.problem.mode_details[t][m].get(resource, 0) > 0
        ]
        capa = self.problem.get_max_resource_capacity(resource)
        self.cp_model.add(
            sum([q * self.variables["presences"][t][m] for t, m, q in potential_tasks])
            <= capa
        )

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> Solution:
        schedule = {}
        modes_dict = {}
        for t in self.variables["starts"]:
            st = cpsolvercb.value(self.variables["starts"][t])
            end = cpsolvercb.value(self.variables["ends"][t])
            schedule[t] = {"starts": [st], "ends": [end]}
            for m in self.variables["presences"][t]:
                if cpsolvercb.value(self.variables["presences"][t][m]) > 0:
                    modes_dict[t] = m
        modes = [modes_dict[t] for t in self.problem.tasks_list_non_dummy]
        return PreemptiveRcpspSolution(
            problem=self.problem, rcpsp_schedule=schedule, rcpsp_modes=modes
        )


def transform_calendar_preemptive_solution_to_preemptive(
    solution: PreemptiveRcpspSolution,
    problem: PreemptiveRcpspProblem,
    resource_calendar_dict: dict[tuple, np.ndarray] = None,
    task_mode_to_calendar: dict[tuple, np.ndarray] = None,
) -> PreemptiveRcpspSolution:
    if resource_calendar_dict is None:
        _, resource_calendar_dict, task_mode_to_calendar = (
            compute_binary_calendar_per_tasks(problem)
        )
    sched = {}
    for t in solution.rcpsp_schedule:
        mode = 1
        if t in problem.tasks_list_non_dummy:
            mode = solution.rcpsp_modes[problem.index_task_non_dummy[t]]
        if (t, mode) not in task_mode_to_calendar:
            sched[t] = solution.rcpsp_schedule[t]
            continue
        calendar = resource_calendar_dict[task_mode_to_calendar[(t, mode)]]
        sts = solution.get_start_times_list(t)
        ends = solution.get_end_times_list(t)
        sched_list = []
        for st, end in zip(sts, ends):
            sched_list += [j for j in range(st, end) if calendar[j]]
        subparts = []
        cur_start = sched_list[0]
        for index in range(1, len(sched_list)):
            if sched_list[index] > sched_list[index - 1] + 1:
                subparts.append((cur_start, sched_list[index - 1] + 1))
                cur_start = sched_list[index]
        subparts.append((cur_start, sched_list[-1] + 1))
        sched[t] = {}
        sched[t]["starts"] = [x[0] for x in subparts]
        sched[t]["ends"] = [x[1] for x in subparts]
    return PreemptiveRcpspSolution(
        problem=problem, rcpsp_schedule=sched, rcpsp_modes=solution.rcpsp_modes
    )


def compute_duration_function_time_cluster(
    orig_duration: int,
    resource_calendar: np.ndarray,
    cumulative_resource_calendar: np.ndarray,
):
    duration = -np.ones((cumulative_resource_calendar.shape[0]))
    dict_of_interval_per_duration = {}
    current_interval = [0, 0]
    cur_duration = -1
    for i in range(cumulative_resource_calendar.shape[0]):
        if resource_calendar[i] == 0:
            if duration[i] == duration[i - 1]:
                current_interval[1] = i
            else:
                prev_d = duration[i - 1]
                if prev_d not in dict_of_interval_per_duration:
                    dict_of_interval_per_duration[prev_d] = []
                dict_of_interval_per_duration[prev_d] += [
                    [current_interval[0], current_interval[1]]
                ]
                current_interval = [i, i]
            continue
        x = cumulative_resource_calendar[i]
        if x == 0:
            continue
        index = next(
            (
                j
                for j in range(i, cumulative_resource_calendar.shape[0])
                if cumulative_resource_calendar[j] == x + orig_duration - 1
            ),
            None,
        )
        if index is not None:
            duration[i] = index - i + 1
            cur_duration = duration[i]
            if i >= 1:
                if duration[i] == duration[i - 1]:
                    current_interval[1] = i
                else:
                    prev_d = duration[i - 1]
                    if prev_d not in dict_of_interval_per_duration:
                        dict_of_interval_per_duration[prev_d] = []
                    dict_of_interval_per_duration[prev_d] += [
                        [current_interval[0], current_interval[1]]
                    ]
                    current_interval = [i, i]
        else:
            break
    if current_interval[0] != current_interval[1]:
        d = cur_duration
        if d not in dict_of_interval_per_duration:
            dict_of_interval_per_duration[d] = []
        dict_of_interval_per_duration[d] += [[current_interval[0], current_interval[1]]]
    if len(dict_of_interval_per_duration) == 0:
        dict_of_interval_per_duration[orig_duration] = current_interval
    # print(dict_of_interval_per_duration)
    return duration, dict_of_interval_per_duration


def compute_binary_calendar_per_tasks(
    problem: PreemptiveRcpspProblem,
) -> tuple[tuple, dict[tuple, np.ndarray], dict[tuple[int, int], tuple]]:
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
                durations[i, m] = compute_duration_function_time_cluster(
                    orig_duration=duration,
                    resource_calendar=c,
                    cumulative_resource_calendar=np.cumsum(c),
                )
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
                durations[i, m] = compute_duration_function_time_cluster(
                    orig_duration=duration,
                    resource_calendar=resource_calendar_dict[tuple_res],
                    cumulative_resource_calendar=cumulative_calendar_dict[tuple_res],
                )
                task_mode_to_calendar[i, m] = tuple_res
    return durations, resource_calendar_dict, task_mode_to_calendar
