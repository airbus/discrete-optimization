#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Any

from discrete_optimization.generic_tasks_tools.solvers.optalcp.allocation import (
    AllocationOptalSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.optalcp.scheduling import (
    SchedulingOptalSolver,
)
from discrete_optimization.rcalbp_l.problem import (
    RCALBPLProblem,
    RCALBPLSolution,
    Task,
    WorkStation,
)

try:
    import optalcp as cp
except ImportError:
    pass


class OptalRCALBPLSolver(
    AllocationOptalSolver[Task, WorkStation], SchedulingOptalSolver[Task]
):
    problem: RCALBPLProblem
    variables: dict

    def init_model(self, **kwargs: Any) -> None:
        super().init_model(**kwargs)
        self.cp_model = cp.Model(name="RCALBPL")
        self.variables = {}

        # 1. Create variables
        self.create_main_unfolded_intervals()
        self.create_resource_dispatch()
        self.create_cycle_time_variables()
        # 2. Post constraints
        self.constraint_only_one_station_allocation()
        self.create_cumulative_resource_constraint()
        self.create_zone_blocking()
        self.create_precedence_constraints()
        # self.create_heuristic_target_reached_constraints(apply_heuristic=False)
        # 3. Post objective
        self.objective_value()

    def create_main_unfolded_intervals(self):
        dict_main_intervals = {}
        dict_opt_intervals = {}
        allocations = {}
        max_horizon = self.problem.c_max

        # Iterate over the periods list (not just integer)
        for p in self.problem.periods:
            # Unfold time ONLY by period. All workstations run in parallel within this window.
            p_lb_start = 0
            p_ub_start = max_horizon
            for task in self.problem.tasks:
                possible_durations = [
                    self.problem.get_duration(task, p, w) for w in self.problem.stations
                ]
                min_dur = min(possible_durations)
                max_dur = max(possible_durations)
                dict_main_intervals[(task, p)] = self.cp_model.interval_var(
                    start=(p_lb_start, p_ub_start),
                    end=(p_lb_start, p_ub_start),
                    length=(min_dur, max_dur),
                    optional=False,
                    name=f"{task}_{p}_interval_unfolded",
                )
                for w in self.problem.stations:
                    dur = self.problem.get_duration(task, p, w)
                    # Opt intervals share the exact same time window allowing parallel workstation overlap
                    opt_var = self.cp_model.interval_var(
                        start=(p_lb_start, p_ub_start),
                        end=(p_lb_start, p_ub_start),
                        length=dur,
                        optional=True,
                        name=f"{task}_{p}_{w}_interval_unfolded",
                    )
                    dict_opt_intervals[(task, p, w)] = opt_var
                    # Populate allocations dictionary with presence variables
                    allocations[(task, p, w)] = self.cp_model.presence(opt_var)
                    if p >= 1:
                        self.cp_model.enforce(
                            allocations[(task, p, w)] == allocations[(task, 0, w)]
                        )
                # Link main interval with its workstation alternatives
                self.cp_model.alternative(
                    dict_main_intervals[(task, p)],
                    [dict_opt_intervals[(task, p, w)] for w in self.problem.stations],
                )

        self.variables["main_intervals"] = dict_main_intervals
        self.variables["opt_intervals"] = dict_opt_intervals
        self.variables["allocations"] = allocations

    def constraint_only_one_station_allocation(self):
        for task in self.problem.tasks:
            is_allocated = []
            for w in self.problem.stations:
                allocated_to_station = self.cp_model.max(
                    [
                        self.variables["allocations"][(task, p, w)]
                        for p in self.problem.periods
                    ]
                )
                is_allocated.append(allocated_to_station)
            self.cp_model.enforce(self.cp_model.sum(is_allocated) == 1)

    def create_resource_dispatch(self):
        resource = {}
        for r in self.problem.resources:
            capa = self.problem.capa_resources[r]
            for w in self.problem.stations:
                resource[(r, w)] = self.cp_model.int_var(
                    min=0, max=capa, name=f"capa_{r}_station_{w}"
                )
            self.cp_model.enforce(
                self.cp_model.sum([resource[r, w] for w in self.problem.stations])
                <= capa
            )
        self.variables["resource_dispatch"] = resource

    def create_cumulative_resource_constraint(self):
        max_horizon = self.problem.c_max

        for r in self.problem.resources:
            capa = self.problem.capa_resources[r]
            for p in self.problem.periods:
                # Time bounds isolating this period
                p_lb = 0
                p_ub = max_horizon
                # 1. Global capacity bound across all workstations
                pulses_global = []
                for task in self.problem.tasks:
                    req = self.problem.cons_resources[r][task]  # Indexing is [r][task]
                    if req > 0:
                        pulses_global.append(
                            self.cp_model.pulse(
                                self.variables["main_intervals"][(task, p)], req
                            )
                        )
                if pulses_global:
                    self.cp_model.enforce(self.cp_model.sum(pulses_global) <= capa)
                # 2. Local capacity bound per workstation (dispatch limit)
                for w in self.problem.stations:
                    pulses_local = []
                    for task in self.problem.tasks:
                        req = self.problem.cons_resources[r][task]
                        if req > 0:
                            pulses_local.append(
                                self.cp_model.pulse(
                                    self.variables["opt_intervals"][(task, p, w)], req
                                )
                            )

                    # Dummy interval trick: Consumes the unallocated portion of the resource dispatch
                    unallocated = capa - self.variables["resource_dispatch"][(r, w)]
                    dummy_interval = self.cp_model.interval_var(
                        start=p_lb, end=p_ub, length=max_horizon, optional=False
                    )
                    pulses_local.append(
                        self.cp_model.pulse(dummy_interval, unallocated)
                    )
                    self.cp_model.enforce(self.cp_model.sum(pulses_local) <= capa)
        # 3. Zone capacities (local to workstations)
        for z in self.problem.zones:
            capa = self.problem.capa_zones[z]
            for p in self.problem.periods:
                for w in self.problem.stations:  # Zones are evaluated per workstation
                    pulses_zone = []
                    for task in self.problem.tasks:
                        req = self.problem.cons_zones[z][task]  # Indexing [z][task]
                        if req > 0:
                            pulses_zone.append(
                                self.cp_model.pulse(
                                    self.variables["opt_intervals"][(task, p, w)], req
                                )
                            )
                    if pulses_zone:
                        self.cp_model.enforce(self.cp_model.sum(pulses_zone) <= capa)

    def create_zone_blocking(self):
        for z in self.problem.zones:
            tasks_blocking = [
                t for t in self.problem.tasks if z in self.problem.neutr_zones[t]
            ]
            tasks_consuming = [
                t
                for t in self.problem.tasks
                if self.problem.cons_zones[z][t] > 0
                if t not in tasks_blocking
            ]
            # Since the zones have a capacity of 1 this sum threshold logic works perfectly
            if self.problem.capa_zones[z] == 1 and tasks_blocking:
                for p in self.problem.periods:
                    for w in self.problem.stations:
                        pulses = [
                            self.cp_model.pulse(
                                self.variables["opt_intervals"][(t, p, w)], 1
                            )
                            for t in tasks_blocking
                        ]
                        pulses.extend(
                            [
                                self.cp_model.pulse(
                                    self.variables["opt_intervals"][(t, p, w)],
                                    len(tasks_blocking),
                                )
                                for t in tasks_consuming
                            ]
                        )

                        if pulses:
                            self.cp_model.enforce(
                                self.cp_model.sum(pulses) <= len(tasks_blocking)
                            )

    def create_precedence_constraints(self):
        for t1, t2 in self.problem.precedences:
            # 1. Global Station Precedence: wks(t1) <= wks(t2)
            wks_t1 = self.cp_model.sum(
                [
                    w * self.variables["allocations"][(t1, 0, w)]
                    for w in self.problem.stations
                ]
            )
            wks_t2 = self.cp_model.sum(
                [
                    w * self.variables["allocations"][(t2, 0, w)]
                    for w in self.problem.stations
                ]
            )
            self.cp_model.enforce(wks_t1 <= wks_t2)

            # 2. Temporal Precedence: Valid ONLY if they share the same workstation
            for p in self.problem.periods:
                for w in self.problem.stations:
                    self.cp_model.end_before_start(
                        self.variables["opt_intervals"][(t1, p, w)],
                        self.variables["opt_intervals"][(t2, p, w)],
                    )

    def create_cycle_time_variables(self):
        cycle_time_used = {}
        cycle_time_chosen = {}
        max_horizon = self.problem.c_max

        for p in self.problem.periods:
            cycle_time_chosen[p] = self.cp_model.int_var(
                min=self.problem.c_target,
                max=max_horizon,
                name=f"cycle_time_chosen_{p}",
            )
            cycle_time_used[p] = self.cp_model.int_var(
                min=0, max=max_horizon, name=f"cycle_time_used_{p}"
            )
            # Baseline offset applied for this specific period's unfolded timeline
            max_ends = []
            for task in self.problem.tasks:
                main_int = self.variables["main_intervals"][(task, p)]
                # Since all active tasks inside main_intervals perfectly overlap
                # within [p*max, (p+1)*max], we just subtract init_time to get the relative end.
                relative_end = self.cp_model.end(main_int)
                max_ends.append(relative_end)
            cycle_time_used[p] = self.cp_model.max(max_ends)
            # self.cp_model.enforce(cycle_time_used[p] == self.cp_model.max(max_ends))
            self.cp_model.enforce(cycle_time_chosen[p] >= cycle_time_used[p])

        self.variables["cycle_time_used"] = cycle_time_used
        self.variables["cycle_time_chosen"] = cycle_time_chosen

        # Unstable periods logic: Cycle time remains constant [cite: 176, 178, 439]
        if self.problem.nb_stations > 0:
            self.cp_model.enforce(
                self.variables["cycle_time_chosen"][0]
                == self.variables["cycle_time_used"][0]
            )
            for p in range(1, self.problem.nb_stations):
                self.cp_model.enforce(
                    self.variables["cycle_time_chosen"][p]
                    == self.variables["cycle_time_chosen"][p - 1]
                )
                self.cp_model.enforce(
                    self.variables["cycle_time_used"][p]
                    == self.variables["cycle_time_used"][p - 1]
                )

        # Stable periods logic: Cycle time is monotonically decreasing [cite: 488, 499]
        for p in range(self.problem.nb_stations, self.problem.nb_periods):
            self.cp_model.enforce(
                self.variables["cycle_time_chosen"][p]
                <= self.variables["cycle_time_chosen"][p - 1]
            )
            self.cp_model.enforce(
                self.variables["cycle_time_used"][p]
                <= self.variables["cycle_time_used"][p - 1]
            )
            self.cp_model.enforce(
                self.cp_model.implies(
                    self.variables["cycle_time_chosen"][p]
                    < self.variables["cycle_time_chosen"][p - 1],
                    self.variables["cycle_time_chosen"][p]
                    == self.variables["cycle_time_used"][p],
                )
            )

    def objective_value(self):
        obj_terms = []
        for p in self.problem.periods:
            if p < self.problem.nb_stations:
                obj_terms.append(self.variables["cycle_time_chosen"][p])
            else:
                cost = self.cp_model.int_var(
                    min=0, max=self.problem.c_max, name=f"cost_{p}"
                )
                self.cp_model.enforce(
                    self.cp_model.implies(
                        (
                            self.variables["cycle_time_chosen"][p]
                            > self.problem.c_target
                        ),
                        cost == self.variables["cycle_time_chosen"][p],
                    )
                )
                self.cp_model.enforce(
                    self.cp_model.implies(
                        (
                            self.variables["cycle_time_chosen"][p]
                            == self.problem.c_target
                        ),
                        cost == 0,
                    )
                )
                # obj_terms.append(cost)
                obj_terms.append(self.variables["cycle_time_chosen"][p])
        self.cp_model.minimize(self.cp_model.sum(obj_terms))

    def create_heuristic_target_reached_constraints(self, apply_heuristic: bool = True):
        max_horizon = self.problem.c_max
        c_target = self.problem.c_target

        # 1. Constraint: If the "used" cycle time <= c_target, clamp the "chosen" cycle time to c_target.
        for p in self.problem.periods:
            # Link the boolean variable to the condition
            self.cp_model.enforce(
                (self.variables["cycle_time_used"][p] <= c_target)
                == (self.variables["cycle_time_chosen"][p] == c_target)
            )

        # 2. Heuristic: Freeze future schedules once target is reached
        if apply_heuristic:
            # We only apply this starting from the first stable period
            for p in range(self.problem.nb_stations, self.problem.nb_periods - 1):
                for t in self.problem.tasks:
                    # Isolate the relative start time of task t for period p and p+1
                    rel_start_p = self.cp_model.start(
                        self.variables["main_intervals"][(t, p)]
                    )
                    rel_start_next = self.cp_model.start(
                        self.variables["main_intervals"][(t, p + 1)]
                    )
                    # Implication: If target is reached at period p, enforce that the relative
                    # start time in period p+1 is strictly equal to the start time in period p.
                    self.cp_model.enforce(
                        self.cp_model.implies(
                            self.variables["cycle_time_chosen"][p] == c_target,
                            (rel_start_p == rel_start_next),
                        )
                    )

    def get_task_unary_resource_is_present_variable(
        self, task: Task, unary_resource: WorkStation
    ) -> "cp.BoolExpr":
        return self.variables["allocations"][(task[0], task[1], unary_resource)]

    def get_task_interval_variable(self, task: Task) -> "cp.IntervalVar":
        return self.variables["main_intervals"][task]

    def retrieve_solution(self, result: "cp.SolveResult") -> RCALBPLSolution:
        """
        Parses the OptalCP result back into a native RCALBPLSolution, translating
        absolute unfolded times back to relative workstation times.
        """
        wks = {}
        raw = {}
        start = {}
        cyc = {}

        # Retrieve allocations
        for t in self.problem.tasks:
            for w in self.problem.stations:
                # Task allocation is constant, so we can just check period 0
                if result.solution.is_present(
                    self.variables["opt_intervals"][(t, 0, w)]
                ):
                    wks[t] = w
                    break

        # Retrieve resource dispatch bounds
        for r in self.problem.resources:
            for w in self.problem.stations:
                raw[(r, w)] = result.solution.get_value(
                    self.variables["resource_dispatch"][(r, w)]
                )

        # Retrieve cycle times chosen
        for p in self.problem.periods:
            cyc[p] = result.solution.get_value(self.variables["cycle_time_chosen"][p])

        # Retrieve relative task start times
        for p in self.problem.periods:
            for t in self.problem.tasks:
                w = wks[t]
                # Fetch absolute start time and subtract baseline offset
                start[(t, p)] = result.solution.get_start(
                    self.variables["opt_intervals"][(t, p, w)]
                )
        return RCALBPLSolution(
            problem=self.problem, wks=wks, raw=raw, start=start, cyc=cyc
        )


class OptalRCALBPLSolverV2(
    AllocationOptalSolver[Task, WorkStation], SchedulingOptalSolver[Task]
):
    problem: RCALBPLProblem
    variables: dict

    def init_model(self, **kwargs: Any) -> None:
        super().init_model(**kwargs)
        self.cp_model = cp.Model(name="RCALBPL")
        self.variables = {}

        # 1. Create variables
        self.create_main_unfolded_intervals()
        self.create_resource_dispatch()
        self.create_cycle_time_variables()
        # 2. Post constraints
        self.create_cumulative_resource_constraint()
        self.create_zone_blocking()
        self.create_precedence_constraints()
        # self.create_heuristic_target_reached_constraints(apply_heuristic=False)
        # 3. Post objective
        self.objective_value()

    def create_main_unfolded_intervals(self):
        dict_main_intervals = {}
        dict_opt_intervals = {}
        allocations = {}
        max_horizon = self.problem.c_max

        # Iterate over the periods list (not just integer)
        for p in self.problem.periods:
            # Unfold time ONLY by period. All workstations run in parallel within this window.
            p_lb_start = max_horizon * (p * self.problem.nb_stations)
            p_ub_start = max_horizon * ((p + 1) * self.problem.nb_stations)
            for task in self.problem.tasks:
                possible_durations = [
                    self.problem.get_duration(task, p, w) for w in self.problem.stations
                ]
                min_dur = min(possible_durations)
                max_dur = max(possible_durations)
                dict_main_intervals[(task, p)] = self.cp_model.interval_var(
                    start=(p_lb_start, p_ub_start),
                    end=(p_lb_start, p_ub_start),
                    length=(min_dur, max_dur),
                    optional=False,
                    name=f"{task}_{p}_interval_unfolded",
                )
                if p == 0:
                    for w in range(self.problem.nb_stations):
                        allocations[(task, w)] = (
                            self.cp_model.start(dict_main_intervals[(task, p)])
                            >= self.problem.c_max * w
                        ) & (
                            self.cp_model.end(dict_main_intervals[(task, p)])
                            <= self.problem.c_max * (w + 1)
                        )
                    self.cp_model.enforce(
                        self.cp_model.sum(
                            [
                                allocations[(task, w)]
                                for w in range(self.problem.nb_stations)
                            ]
                        )
                        == 1
                    )
                    w = self.cp_model.sum(
                        [
                            w * allocations[(task, w)]
                            for w in range(self.problem.nb_stations)
                        ]
                    )
                    self.cp_model.enforce(
                        self.cp_model.implies(
                            (p - w) >= 0,
                            self.cp_model.length(dict_main_intervals[(task, p)])
                            == self.cp_model.element(
                                self.problem.durations[task], p - w
                            ),
                        )
                    )
                    self.cp_model.enforce(
                        self.cp_model.implies(
                            (p - w) < 0,
                            self.cp_model.length(dict_main_intervals[(task, p)]) == 0,
                        )
                    )
                else:
                    w = self.cp_model.sum(
                        [
                            w * allocations[(task, w)]
                            for w in range(self.problem.nb_stations)
                        ]
                    )
                    self.cp_model.enforce(
                        self.cp_model.implies(
                            (p - w) >= 0,
                            self.cp_model.length(dict_main_intervals[(task, p)])
                            == self.cp_model.element(
                                self.problem.durations[task], p - w
                            ),
                        )
                    )
                    self.cp_model.enforce(
                        self.cp_model.implies(
                            (p - w) < 0,
                            self.cp_model.length(dict_main_intervals[(task, p)]) == 0,
                        )
                    )
                    self.cp_model.enforce(
                        self.cp_model.start(dict_main_intervals[(task, p)])
                        >= (self.problem.nb_stations * p + w) * max_horizon
                    )
                    self.cp_model.enforce(
                        self.cp_model.end(dict_main_intervals[(task, p)])
                        <= (self.problem.nb_stations * p + w + 1) * max_horizon
                    )
        self.cp_model.enforce(
            self.cp_model.sum([allocations[task, 1] for task in self.problem.tasks])
            >= 30
        )
        self.variables["main_intervals"] = dict_main_intervals
        self.variables["opt_intervals"] = dict_opt_intervals
        self.variables["allocations"] = allocations

    def create_resource_dispatch(self):
        resource = {}
        for r in self.problem.resources:
            capa = self.problem.capa_resources[r]
            for w in self.problem.stations:
                resource[(r, w)] = self.cp_model.int_var(
                    min=0, max=capa, name=f"capa_{r}_station_{w}"
                )
            self.cp_model.enforce(
                self.cp_model.sum([resource[r, w] for w in self.problem.stations])
                <= capa
            )
        self.variables["resource_dispatch"] = resource

    def create_cumulative_resource_constraint(self):
        max_horizon = self.problem.c_max

        for r in self.problem.resources:
            capa = self.problem.capa_resources[r]
            pulses_resources = []
            pulses_global = []
            for p in self.problem.periods:
                for w in self.problem.stations:
                    lb = (p * self.problem.nb_stations + w) * max_horizon
                    ub = (p * self.problem.nb_stations + w + 1) * max_horizon
                    pulses_resources.append(
                        self.cp_model.pulse(
                            self.cp_model.interval_var(start=lb, end=ub),
                            capa - self.variables["resource_dispatch"][r, w],
                        )
                    )
                # 1. Global capacity bound across all workstations
                for task in self.problem.tasks:
                    req = self.problem.cons_resources[r][task]  # Indexing is [r][task]
                    if req > 0:
                        pulses_global.append(
                            self.cp_model.pulse(
                                self.variables["main_intervals"][(task, p)], req
                            )
                        )
            if pulses_global:
                self.cp_model.enforce(
                    self.cp_model.sum(pulses_global + pulses_resources) <= capa
                )
        # 3. Zone capacities (local to workstations)
        for z in self.problem.zones:
            capa = self.problem.capa_zones[z]
            for p in self.problem.periods:
                pulses_zone = []
                for task in self.problem.tasks:
                    req = self.problem.cons_zones[z][task]  # Indexing [z][task]
                    if req > 0:
                        pulses_zone.append(
                            self.cp_model.pulse(
                                self.variables["main_intervals"][(task, p)], req
                            )
                        )
                if pulses_zone:
                    self.cp_model.enforce(self.cp_model.sum(pulses_zone) <= capa)

    def create_zone_blocking(self):
        for z in self.problem.zones:
            tasks_blocking = [
                t for t in self.problem.tasks if z in self.problem.neutr_zones[t]
            ]
            tasks_consuming = [
                t
                for t in self.problem.tasks
                if self.problem.cons_zones[z][t] > 0
                if t not in tasks_blocking
            ]
            # Since the zones have a capacity of 1 this sum threshold logic works perfectly
            if self.problem.capa_zones[z] == 1 and tasks_blocking:
                for p in self.problem.periods:
                    pulses = [
                        self.cp_model.pulse(self.variables["main_intervals"][(t, p)], 1)
                        for t in tasks_blocking
                    ]
                    pulses.extend(
                        [
                            self.cp_model.pulse(
                                self.variables["main_intervals"][(t, p)],
                                len(tasks_blocking),
                            )
                            for t in tasks_consuming
                        ]
                    )
                    if pulses:
                        self.cp_model.enforce(
                            self.cp_model.sum(pulses) <= len(tasks_blocking)
                        )

    def create_precedence_constraints(self):
        for t1, t2 in self.problem.precedences:
            # 2. Temporal Precedence: Valid ONLY if they share the same workstation
            for p in self.problem.periods:
                self.cp_model.end_before_start(
                    self.variables["main_intervals"][(t1, p)],
                    self.variables["main_intervals"][(t2, p)],
                )

    def create_cycle_time_variables(self):
        cycle_time_used = {}
        cycle_time_chosen = {}
        max_horizon = self.problem.c_max
        for p in self.problem.periods:
            cycle_time_chosen[p] = self.cp_model.int_var(
                min=self.problem.c_target,
                max=max_horizon,
                name=f"cycle_time_chosen_{p}",
            )
            cycle_time_used[p] = self.cp_model.int_var(
                min=0, max=max_horizon, name=f"cycle_time_used_{p}"
            )
            # Baseline offset applied for this specific period's unfolded timeline
            max_ends = []
            for task in self.problem.tasks:
                allocation = self.cp_model.sum(
                    [
                        k * self.variables["allocations"][task, k]
                        for k in range(self.problem.nb_stations)
                    ]
                )
                init_time = (
                    self.problem.nb_stations * p + allocation
                ) * self.problem.c_max
                main_int = self.variables["main_intervals"][(task, p)]
                relative_end = self.cp_model.end(main_int) - init_time
                max_ends.append(relative_end)
            cycle_time_used[p] = self.cp_model.max(max_ends)
            # self.cp_model.enforce(cycle_time_used[p] == self.cp_model.max(max_ends))
            self.cp_model.enforce(cycle_time_chosen[p] >= cycle_time_used[p])

        self.variables["cycle_time_used"] = cycle_time_used
        self.variables["cycle_time_chosen"] = cycle_time_chosen

        # Unstable periods logic: Cycle time remains constant [cite: 176, 178, 439]
        if self.problem.nb_stations > 0:
            self.cp_model.enforce(
                self.variables["cycle_time_chosen"][0]
                == self.variables["cycle_time_used"][0]
            )
            for p in range(1, self.problem.nb_stations):
                self.cp_model.enforce(
                    self.variables["cycle_time_chosen"][p]
                    == self.variables["cycle_time_chosen"][p - 1]
                )
                self.cp_model.enforce(
                    self.variables["cycle_time_used"][p]
                    == self.variables["cycle_time_used"][p - 1]
                )

        # Stable periods logic: Cycle time is monotonically decreasing [cite: 488, 499]
        for p in range(self.problem.nb_stations, self.problem.nb_periods):
            self.cp_model.enforce(
                self.variables["cycle_time_chosen"][p]
                <= self.variables["cycle_time_chosen"][p - 1]
            )
            self.cp_model.enforce(
                self.variables["cycle_time_used"][p]
                <= self.variables["cycle_time_used"][p - 1]
            )
            self.cp_model.enforce(
                self.cp_model.implies(
                    self.variables["cycle_time_chosen"][p]
                    < self.variables["cycle_time_chosen"][p - 1],
                    self.variables["cycle_time_chosen"][p]
                    == self.variables["cycle_time_used"][p],
                )
            )

    def objective_value(self):
        obj_terms = []
        for p in self.problem.periods:
            if p < self.problem.nb_stations:
                obj_terms.append(self.variables["cycle_time_chosen"][p])
            else:
                cost = self.cp_model.int_var(
                    min=0, max=self.problem.c_max, name=f"cost_{p}"
                )
                self.cp_model.enforce(
                    self.cp_model.implies(
                        (
                            self.variables["cycle_time_chosen"][p]
                            > self.problem.c_target
                        ),
                        cost == self.variables["cycle_time_chosen"][p],
                    )
                )
                self.cp_model.enforce(
                    self.cp_model.implies(
                        (
                            self.variables["cycle_time_chosen"][p]
                            == self.problem.c_target
                        ),
                        cost == 0,
                    )
                )
                obj_terms.append(cost)
        self.cp_model.minimize(self.cp_model.sum(obj_terms))

    def create_heuristic_target_reached_constraints(self, apply_heuristic: bool = True):
        max_horizon = self.problem.c_max
        c_target = self.problem.c_target

        # 1. Constraint: If the "used" cycle time <= c_target, clamp the "chosen" cycle time to c_target.
        for p in self.problem.periods:
            # Link the boolean variable to the condition
            self.cp_model.enforce(
                (self.variables["cycle_time_used"][p] <= c_target)
                == (self.variables["cycle_time_chosen"][p] == c_target)
            )

        # 2. Heuristic: Freeze future schedules once target is reached
        if False and apply_heuristic:
            # We only apply this starting from the first stable period
            for p in range(self.problem.nb_stations, self.problem.nb_periods - 1):
                for t in self.problem.tasks:
                    # Isolate the relative start time of task t for period p and p+1
                    rel_start_p = self.cp_model.start(
                        self.variables["main_intervals"][(t, p)]
                    ) - (p * max_horizon)
                    rel_start_next = self.cp_model.start(
                        self.variables["main_intervals"][(t, p + 1)]
                    ) - ((p + 1) * max_horizon)

                    # Implication: If target is reached at period p, enforce that the relative
                    # start time in period p+1 is strictly equal to the start time in period p.
                    self.cp_model.enforce(
                        (self.variables["cycle_time_chosen"][p] == c_target)
                        <= (rel_start_p == rel_start_next)
                    )

    def get_task_unary_resource_is_present_variable(
        self, task: Task, unary_resource: WorkStation
    ) -> "cp.BoolExpr":
        return self.variables["allocations"][(task[0], task[1], unary_resource)]

    def get_task_interval_variable(self, task: Task) -> "cp.IntervalVar":
        return self.variables["main_intervals"][task]

    def retrieve_solution(self, result: "cp.SolveResult") -> RCALBPLSolution:
        """
        Parses the OptalCP result back into a native RCALBPLSolution, translating
        absolute unfolded times back to relative workstation times.
        """
        wks = {}
        raw = {}
        start = {}
        cyc = {}

        # Retrieve allocations
        for t in self.problem.tasks:
            # Task allocation is constant, so we can just check period 0
            start_ = result.solution.get_start(self.variables["main_intervals"][(t, 0)])
            wks[t] = start_ // self.problem.c_max
            print(start_, self.problem.c_max)
            print(wks[t])
        # Retrieve resource dispatch bounds
        for r in self.problem.resources:
            for w in self.problem.stations:
                raw[(r, w)] = result.solution.get_value(
                    self.variables["resource_dispatch"][(r, w)]
                )

        # Retrieve cycle times chosen
        for p in self.problem.periods:
            cyc[p] = result.solution.get_value(self.variables["cycle_time_chosen"][p])

        # Retrieve relative task start times
        for p in self.problem.periods:
            # Baseline offset for period (reflecting the new unfolded timeline)
            for t in self.problem.tasks:
                # Fetch absolute start time and subtract baseline offset
                abs_start = result.solution.get_start(
                    self.variables["main_intervals"][(t, p)]
                )
                start[(t, p)] = abs_start % self.problem.c_max

        return RCALBPLSolution(
            problem=self.problem, wks=wks, raw=raw, start=start, cyc=cyc
        )
