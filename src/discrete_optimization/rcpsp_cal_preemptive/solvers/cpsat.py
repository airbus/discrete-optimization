#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  Custom implementation
from typing import Any

from ortools.sat.python.cp_model import CpSolverSolutionCallback, Domain

from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver
from discrete_optimization.rcpsp.solution import RcpspSolution
from discrete_optimization.rcpsp.solvers.preemptive.cpsat import (
    compute_binary_calendar_per_tasks,
)
from discrete_optimization.rcpsp.utils import create_fake_tasks
from discrete_optimization.rcpsp_cal_preemptive.problem import (
    CalendarPreemptiveRcpspProblem,
)


class CpSatCalendarPreemptiveSolver(OrtoolsCpSatSolver):
    problem: CalendarPreemptiveRcpspProblem

    def __init__(
        self,
        problem: CalendarPreemptiveRcpspProblem,
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
            (t, i, conso)
            for t in self.variables["opt_intervals"]
            for i in self.variables["opt_intervals"][t]
            if (conso := self.problem.mode_details[t][i].get(resource, 0)) > 0
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

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> RcpspSolution:
        schedule = {}
        modes_dict = {}
        for t in self.variables["starts"]:
            st = cpsolvercb.value(self.variables["starts"][t])
            end = cpsolvercb.value(self.variables["ends"][t])
            schedule[t] = {"start_time": st, "end_time": end}
            for m in self.variables["presences"][t]:
                if cpsolvercb.value(self.variables["presences"][t][m]) > 0:
                    modes_dict[t] = m
        modes = [modes_dict[t] for t in self.problem.tasks_list_non_dummy]
        return RcpspSolution(
            problem=self.problem, rcpsp_schedule=schedule, rcpsp_modes=modes
        )
