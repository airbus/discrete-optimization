#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

import logging
from typing import Any, Iterable

from ortools.sat.python.cp_model import (
    CpSolverSolutionCallback,
    Domain,
    LinearExprT,
)

from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.solvers.cpsat import (
    AllocationCpSatSolver,
    SchedulingCpSatSolver,
)
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.rcalbp_l.problem import (
    RCALBPLProblem,
    RCALBPLSolution,
    Task,
    WorkStation,
)

logger = logging.getLogger(__name__)


class CpSatRCALBPLSolver(
    AllocationCpSatSolver[Task, WorkStation],
    SchedulingCpSatSolver[Task],
    WarmstartMixin,
):
    problem: RCALBPLProblem
    variables: dict

    @property
    def subset_tasks_of_interest(self) -> Iterable[Task]:
        """Subset of tasks of interest for the allocation..."""
        return [t for t in self.problem.tasks_list if t[1] == 0]

    def get_task_start_or_end_variable(
        self, task: Task, start_or_end: StartOrEnd
    ) -> LinearExprT:
        if start_or_end == StartOrEnd.START:
            return self.variables["starts"][task]
        elif start_or_end == StartOrEnd.END:
            return self.variables["ends"][task]

    def init_model(
        self,
        minimize_used_cycle_time: bool = False,
        add_heuristic_constraint: bool = True,
        **kwargs: Any,
    ) -> None:
        super().init_model(**kwargs)
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

        # 3. Post objective
        self.objective_value(minimize_used_cycle_time)
        if add_heuristic_constraint:
            self.create_heuristic_target_reached_constraints(True)

    def create_main_unfolded_intervals(self):
        starts = {}
        ends = {}
        durations = {}
        dict_main_intervals = {}
        dict_opt_intervals = {}
        allocations = {}
        max_horizon = self.problem.c_max

        for p in self.problem.periods:
            # Time bounds isolating this period relative to 0
            p_lb_start = 0
            p_ub_start = max_horizon
            for task in self.problem.tasks:
                possible_durations = [
                    self.problem.get_duration(task, p, w) for w in self.problem.stations
                ]

                starts[(task, p)] = self.cp_model.NewIntVar(
                    lb=p_lb_start, ub=p_ub_start, name=f"start_{task}_{p}"
                )
                ends[(task, p)] = self.cp_model.NewIntVar(
                    lb=p_lb_start, ub=p_ub_start, name=f"end_{task}_{p}"
                )
                durations[(task, p)] = self.cp_model.NewIntVarFromDomain(
                    domain=Domain.FromValues(possible_durations),
                    name=f"duration_{task}_{p}",
                )
                dict_main_intervals[(task, p)] = self.cp_model.NewIntervalVar(
                    start=starts[(task, p)],
                    end=ends[(task, p)],
                    size=durations[(task, p)],
                    name=f"interval_{task}_{p}",
                )

                for w in self.problem.stations:
                    # alloc should be fixed.
                    dur = self.problem.get_duration(task, p, w)
                    allocations[(task, p, w)] = self.cp_model.NewBoolVar(
                        name=f"allocation_{task}_{p}_{w}"
                    )

                    opt_var = self.cp_model.NewOptionalFixedSizeIntervalVar(
                        start=starts[(task, p)],
                        size=dur,
                        is_present=allocations[(task, p, w)],
                        name=f"interval_{task}_{p}_{w}",
                    )
                    dict_opt_intervals[(task, p, w)] = opt_var
                    if p > self.problem.periods[0]:
                        self.cp_model.Add(
                            allocations[(task, p, w)]
                            == allocations[(task, self.problem.periods[0], w)]
                        )
                    self.cp_model.add(dur == durations[task, p]).only_enforce_if(
                        allocations[(task, p, w)]
                    )
                self.cp_model.add_exactly_one(
                    [allocations[(task, p, w)] for w in self.problem.stations]
                )
        self.variables["starts"] = starts
        self.variables["ends"] = ends
        self.variables["durations"] = durations
        self.variables["main_intervals"] = dict_main_intervals
        self.variables["opt_intervals"] = dict_opt_intervals
        self.variables["allocations"] = allocations

    def constraint_only_one_station_allocation(self):
        for task in self.problem.tasks:
            for w in self.problem.stations:
                allocated_to_station = [
                    self.variables["allocations"][task, p, w]
                    for p in self.problem.periods
                ]
                for i in range(1, len(allocated_to_station)):
                    self.cp_model.add(
                        allocated_to_station[i] == allocated_to_station[i - 1]
                    )

    def create_resource_dispatch(self):
        resource = {}
        for r in self.problem.resources:
            capa = self.problem.capa_resources[r]
            for w in self.problem.stations:
                resource[(r, w)] = self.cp_model.NewIntVar(
                    lb=0, ub=capa, name=f"capa_{r}_station_{w}"
                )
            self.cp_model.add(
                sum([resource[r, w] for w in self.problem.stations]) <= capa
            )
        self.variables["resource_dispatch"] = resource

    def create_cumulative_resource_constraint(self):
        for r in self.problem.resources:
            capa = self.problem.capa_resources[r]
            for p in self.problem.periods:
                # 1. Global capacity bound across all workstations
                pulses_global = []
                for task in self.problem.tasks:
                    req = self.problem.cons_resources[r][task]
                    if req > 0:
                        pulses_global.append(
                            (self.variables["main_intervals"][(task, p)], req)
                        )
                if pulses_global:
                    self.cp_model.add_cumulative(
                        intervals=[x[0] for x in pulses_global],
                        demands=[x[1] for x in pulses_global],
                        capacity=capa,
                    )

                # 2. Local capacity bound per workstation (dispatch limit)
                for w in self.problem.stations:
                    pulses_local = []
                    for task in self.problem.tasks:
                        req = self.problem.cons_resources[r][task]
                        if req > 0:
                            pulses_local.append(
                                (self.variables["opt_intervals"][(task, p, w)], req)
                            )
                    if pulses_local:
                        self.cp_model.add_cumulative(
                            [x[0] for x in pulses_local],
                            [x[1] for x in pulses_local],
                            self.variables["resource_dispatch"][(r, w)],
                        )

        # 3. Zone capacities (local to workstations)
        for z in self.problem.zones:
            capa = self.problem.capa_zones[z]
            for p in self.problem.periods:
                for w in self.problem.stations:
                    pulses_zone = []
                    for task in self.problem.tasks:
                        req = self.problem.cons_zones[z][task]
                        if req > 0:
                            pulses_zone.append(
                                (self.variables["opt_intervals"][(task, p, w)], req)
                            )
                    if pulses_zone:
                        if capa == 1:
                            self.cp_model.add_no_overlap([x[0] for x in pulses_zone])
                        else:
                            self.cp_model.add_cumulative(
                                [x[0] for x in pulses_zone],
                                [x[1] for x in pulses_zone],
                                capa,
                            )

    def create_zone_blocking(self):
        """Be careful on the fact that this constraint only is valid with capa==1 for now"""
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

            if self.problem.capa_zones[z] == 1 and tasks_blocking:
                for p in self.problem.periods:
                    for w in self.problem.stations:
                        pulses = [
                            (self.variables["opt_intervals"][(t, p, w)], 1)
                            for t in tasks_blocking
                        ]
                        pulses.extend(
                            [
                                (
                                    self.variables["opt_intervals"][(t, p, w)],
                                    len(tasks_blocking),
                                )
                                for t in tasks_consuming
                            ]
                        )

                        if pulses:
                            self.cp_model.add_cumulative(
                                [x[0] for x in pulses],
                                [x[1] for x in pulses],
                                len(tasks_blocking),
                            )

    def create_precedence_constraints(self):
        self.variables["same_station"] = {}
        for t1, t2 in self.problem.precedences:
            wks_t1 = sum(
                [
                    w * self.variables["allocations"][(t1, self.problem.periods[0], w)]
                    for w in self.problem.stations
                ]
            )
            wks_t2 = sum(
                [
                    w * self.variables["allocations"][(t2, self.problem.periods[0], w)]
                    for w in self.problem.stations
                ]
            )
            self.cp_model.add(wks_t1 <= wks_t2)

            same_station = self.cp_model.NewBoolVar(name=f"same_station_{t1}_{t2}")
            self.variables["same_station"][(t1, t2)] = same_station

            self.cp_model.add(wks_t1 == wks_t2).only_enforce_if(same_station)
            self.cp_model.add(wks_t2 != wks_t1).only_enforce_if(~same_station)

            for p in self.problem.periods:
                self.cp_model.add(
                    self.variables["ends"][t1, p] <= self.variables["starts"][t2, p]
                ).only_enforce_if(same_station)

    def create_cycle_time_variables(self):
        cycle_time_used = {}
        cycle_time_chosen = {}
        self.variables["is_decreasing"] = {}
        self.variables["is_lower_than_ctarget"] = {}
        max_horizon = self.problem.c_max

        for p in self.problem.periods:
            cycle_time_chosen[p] = self.cp_model.new_int_var(
                lb=self.problem.c_target, ub=max_horizon, name=f"cycle_time_chosen_{p}"
            )
            cycle_time_used[p] = self.cp_model.new_int_var(
                lb=0, ub=max_horizon, name=f"cycle_time_used_{p}"
            )
            for task in self.problem.tasks:
                self.cp_model.add(cycle_time_used[p] >= self.variables["ends"][task, p])
            self.cp_model.add(cycle_time_chosen[p] >= cycle_time_used[p])
            self.variables["is_lower_than_ctarget"][p] = self.cp_model.new_bool_var(
                f"is_lower_than_ctarget_{p}"
            )
            self.cp_model.add(
                cycle_time_used[p] <= self.problem.c_target
            ).only_enforce_if(self.variables["is_lower_than_ctarget"][p])
            self.cp_model.add(
                cycle_time_used[p] > self.problem.c_target
            ).only_enforce_if(~self.variables["is_lower_than_ctarget"][p])
            if p >= self.problem.nb_stations:
                self.cp_model.add(
                    cycle_time_chosen[p] == self.problem.c_target
                ).only_enforce_if(self.variables["is_lower_than_ctarget"][p])
        self.variables["cycle_time_used"] = cycle_time_used
        self.variables["cycle_time_chosen"] = cycle_time_chosen

        unstable_periods = [
            p for p in self.problem.periods if p < self.problem.nb_stations
        ]
        if len(unstable_periods) > 0:
            self.cp_model.add_max_equality(
                self.variables["cycle_time_chosen"][unstable_periods[0]],
                [self.variables["cycle_time_used"][uns] for uns in unstable_periods],
            )
            # Stability.
            for i in range(1, len(unstable_periods)):
                p = unstable_periods[i]
                prev_p = unstable_periods[i - 1]
                self.cp_model.add(
                    self.variables["cycle_time_chosen"][p]
                    == self.variables["cycle_time_chosen"][prev_p]
                )

        # Stable periods logic: Cycle time is monotonically decreasing
        stable_periods = [
            p for p in self.problem.periods if p >= self.problem.nb_stations
        ]
        for p in stable_periods:
            prev_p = p - 1
            if (
                prev_p in self.problem.periods
            ):  # Only link if previous period is also in the modeled window!
                self.cp_model.add(
                    self.variables["cycle_time_chosen"][p]
                    <= self.variables["cycle_time_chosen"][prev_p]
                )
                self.cp_model.add(
                    self.variables["cycle_time_used"][p]
                    <= self.variables["cycle_time_used"][prev_p]
                )

                is_decreasing = self.cp_model.NewBoolVar(name=f"is_decreasing_{p}")
                self.variables["is_decreasing"][p] = is_decreasing

                self.cp_model.add(
                    self.variables["cycle_time_chosen"][p]
                    < self.variables["cycle_time_chosen"][prev_p]
                ).only_enforce_if(is_decreasing)
                self.cp_model.add(
                    self.variables["cycle_time_chosen"][p]
                    >= self.variables["cycle_time_chosen"][prev_p]
                ).only_enforce_if(~is_decreasing)
                self.cp_model.add(
                    self.variables["cycle_time_chosen"][p]
                    == self.variables["cycle_time_used"][p]
                ).only_enforce_if(is_decreasing)

    def fix_allocations(self, wks: dict):
        """
        Hard-fixes the allocation variables to a known configuration.
        This transforms the complex ALBP problem into a scheduling problem.
        """
        if self.cp_model is None:
            return
        for t, w_assigned in wks.items():
            for p in self.problem.periods:
                for w in self.problem.stations:
                    val = 1 if w == w_assigned else 0
                    self.cp_model.add(self.variables["allocations"][(t, p, w)] == val)

    def objective_value(self, minimize_used_cycle_time: bool = False):
        if minimize_used_cycle_time:
            # ALTERNATIVE OBJECTIVE: Squeeze the layout as tight as mathematically possible.
            # Heavily weight 'cycle_time_used' to minimize the actual maximum end time.
            # Lightly weight 'cycle_time_chosen' to prevent floating values.
            obj_terms = []
            for p in self.problem.periods:
                obj_terms.append(self.variables["cycle_time_used"][p])
            self.cp_model.minimize(sum(obj_terms))

        else:
            # ORIGINAL OBJECTIVE: Minimizing Ramp-up cost
            obj_terms = []
            self.variables["cost"] = {}
            self.variables["is_upper_than_ctarget"] = {}

            for p in self.problem.periods:
                if p < self.problem.nb_stations:
                    obj_terms.append(self.variables["cycle_time_chosen"][p])
                else:
                    cost = self.cp_model.NewIntVar(
                        lb=0, ub=self.problem.c_max, name=f"cost_{p}"
                    )
                    is_upper_than_ctarget = self.cp_model.NewBoolVar(
                        name=f"is_upper_than_ctarget_{p}"
                    )

                    self.variables["cost"][p] = cost
                    self.variables["is_upper_than_ctarget"][p] = is_upper_than_ctarget

                    self.cp_model.add(
                        self.variables["cycle_time_chosen"][p] > self.problem.c_target
                    ).only_enforce_if(is_upper_than_ctarget)
                    self.cp_model.add(
                        self.variables["cycle_time_chosen"][p] == self.problem.c_target
                    ).only_enforce_if(~is_upper_than_ctarget)
                    self.cp_model.add(
                        cost == self.variables["cycle_time_chosen"][p]
                    ).only_enforce_if(is_upper_than_ctarget)
                    self.cp_model.add(cost == 0).only_enforce_if(~is_upper_than_ctarget)
                    obj_terms.append(cost)

            self.cp_model.minimize(sum(obj_terms))

    def create_heuristic_target_reached_constraints(self, apply_heuristic: bool = True):
        max_horizon = self.problem.c_max
        c_target = self.problem.c_target

        # 1. Constraint: If the "used" cycle time <= c_target, clamp the "chosen" cycle time to c_target.
        for p in self.problem.periods:
            if p in self.variables["is_upper_than_ctarget"]:
                self.cp_model.add(
                    (self.variables["cycle_time_chosen"][p] == c_target)
                ).only_enforce_if(~self.variables["is_upper_than_ctarget"][p])

        # 2. Heuristic: Freeze future schedules once target is reached
        if apply_heuristic:
            # We only apply this starting from the first stable period
            periods_after_setup = [
                p for p in self.problem.periods if p >= self.problem.nb_stations
            ][:-1]
            for p in periods_after_setup:
                for t in self.problem.tasks:
                    # Isolate the relative start time of task t for period p and p+1
                    start_p = self.variables["starts"][(t, p)]
                    start_next = self.variables["starts"][(t, p + 1)]
                    # Implication: If target is reached at period p, enforce that the relative
                    # start time in period p+1 is strictly equal to the start time in period p.
                    self.cp_model.add(start_p == start_next).only_enforce_if(
                        ~self.variables["is_upper_than_ctarget"][p]
                    )

    def get_task_unary_resource_is_present_variable(
        self, task: Task, unary_resource: WorkStation
    ) -> "cp.BoolExpr":
        return self.variables["allocations"][(task[0], task[1], unary_resource)]

    def retrieve_solution(
        self, cpsolvercb: CpSolverSolutionCallback
    ) -> RCALBPLSolution:
        """
        Parses the CP-SAT result back into a native RCALBPLSolution.
        """
        wks = {}
        raw = {}
        start = {}
        cyc = {}

        # Retrieve allocations
        for t in self.problem.tasks:
            for w in self.problem.stations:
                if cpsolvercb.Value(
                    self.variables["allocations"][(t, self.problem.periods[0], w)]
                ):
                    wks[t] = w
                    break

        # Retrieve resource dispatch bounds
        for r in self.problem.resources:
            for w in self.problem.stations:
                raw[(r, w)] = cpsolvercb.Value(
                    self.variables["resource_dispatch"][(r, w)]
                )

        # Retrieve cycle times chosen
        for p in self.problem.periods:
            cyc[p] = cpsolvercb.Value(self.variables["cycle_time_chosen"][p])

        # Retrieve relative task start times
        for p in self.problem.periods:
            for t in self.problem.tasks:
                start[(t, p)] = cpsolvercb.Value(self.variables["starts"][(t, p)])

        return RCALBPLSolution(
            problem=self.problem, wks=wks, raw=raw, start=start, cyc=cyc
        )

    def set_warm_start(self, solution: RCALBPLSolution) -> None:
        """
        Injects an almost complete model state as a hint to the CP-SAT solver.
        """
        if self.cp_model is None:
            return
        self.cp_model.clear_hints()
        for key in self.allocation_changes_variables:
            self.cp_model.add_hint(self.allocation_changes_variables[key], 0)
        # 1. Hint Allocations
        for t in self.problem.tasks:
            w_assigned = solution.wks.get(t, -1)
            for p in self.problem.periods:
                for w in self.problem.stations:
                    if w == w_assigned:
                        self.cp_model.AddHint(
                            self.variables["allocations"][(t, p, w)], 1
                        )
                    else:
                        self.cp_model.AddHint(
                            self.variables["allocations"][(t, p, w)], 0
                        )

        # 2. Hint Precedence booleans (same_station)
        if "same_station" in self.variables:
            for t1, t2 in self.problem.precedences:
                w1 = solution.wks.get(t1, -1)
                w2 = solution.wks.get(t2, -2)
                is_same = 1 if w1 == w2 else 0
                self.cp_model.AddHint(self.variables["same_station"][(t1, t2)], is_same)

        # 3. Hint Resource Dispatch
        for r in self.problem.resources:
            for w in self.problem.stations:
                val = solution.raw.get((r, w), 0)
                self.cp_model.AddHint(self.variables["resource_dispatch"][(r, w)], val)

        # 4. Hint Cycle times, Task Starts, Ends, Durations, and Objective Flags
        for p in self.problem.periods:
            cyc_curr = solution.cyc.get(p, self.problem.c_max)
            self.cp_model.AddHint(self.variables["cycle_time_chosen"][p], cyc_curr)

            max_end = 0
            for t in self.problem.tasks:
                if (t, p) in solution.start:
                    st = solution.start[(t, p)]
                    w = solution.wks.get(t, 0)
                    dur = self.problem.get_duration(t, p, w)
                    et = st + dur

                    self.cp_model.AddHint(self.variables["starts"][(t, p)], st)
                    self.cp_model.AddHint(self.variables["ends"][(t, p)], et)
                    self.cp_model.AddHint(self.variables["durations"][(t, p)], dur)

                    if et > max_end:
                        max_end = et

            if p in self.variables.get("cycle_time_used", {}):
                self.cp_model.AddHint(self.variables["cycle_time_used"][p], max_end)

            # Hint for is_decreasing
            if p >= self.problem.nb_stations and "is_decreasing" in self.variables:
                prev_cyc = solution.cyc.get(p - 1, self.problem.c_max)
                is_dec = 1 if cyc_curr < prev_cyc else 0
                if p in self.variables["is_decreasing"]:
                    self.cp_model.AddHint(self.variables["is_decreasing"][p], is_dec)

            # Hint for cost and is_upper_than_ctarget
            if p >= self.problem.nb_stations:
                is_upper = 1 if cyc_curr > self.problem.c_target else 0
                cost_val = cyc_curr if is_upper else 0

                if "is_upper_than_ctarget" in self.variables:
                    self.cp_model.AddHint(
                        self.variables["is_upper_than_ctarget"][p], is_upper
                    )
                if "cost" in self.variables:
                    self.cp_model.AddHint(self.variables["cost"][p], cost_val)

    def fix_allocations_and_resources(self, wks: dict, raw: dict):
        """
        Hard-fixes the allocation and resource variables to a known configuration.
        This transforms the complex ALBP problem into a purely independent scheduling problem.
        """
        if getattr(self, "cp_model", None) is None:
            return

        # Lock Task Allocations
        for t, w_assigned in wks.items():
            for p in self.problem.periods:
                for w in self.problem.stations:
                    val = 1 if w == w_assigned else 0
                    self.cp_model.Add(self.variables["allocations"][(t, p, w)] == val)

        # Lock Resource Dispatch
        if "resource_dispatch" in self.variables:
            for (r, w), val in raw.items():
                self.cp_model.Add(self.variables["resource_dispatch"][(r, w)] == val)

    def add_cycle_time_lower_bound(self, p: int, lower_bound: int):
        """
        Ensures cycle time monotonicity across independent chunk boundaries.
        """
        if (
            "cycle_time_chosen" in self.variables
            and p in self.variables["cycle_time_chosen"]
        ):
            self.cp_model.Add(self.variables["cycle_time_chosen"][p] >= lower_bound)
