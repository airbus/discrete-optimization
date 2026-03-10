#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
# Implementation of the problem described in
# https://drops.dagstuhl.de/storage/00lipics/lipics-vol340-cp2025/LIPIcs.CP.2025.25/LIPIcs.CP.2025.25.pdf
import json
from typing import Dict, List, Optional, Tuple

from discrete_optimization.generic_tasks_tools.allocation import (
    AllocationSolution,
)
from discrete_optimization.generic_tasks_tools.scheduling import (
    SchedulingSolution,
)
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveDoc,
    ObjectiveHandling,
    ObjectiveRegister,
    Problem,
    Solution,
    TypeObjective,
)

TaskUnit = int
Period = int
Task = Tuple[TaskUnit, Period]
ResourceCumulative = int
WorkStation = int


class RCALBPLSolution(AllocationSolution[Task, WorkStation], SchedulingSolution[Task]):
    """
    Solution representation for the RC-ALBP/L problem.
    """

    problem: "RCALBPLProblem"

    def __init__(
        self,
        problem: "RCALBPLProblem",
        wks: Dict[TaskUnit, WorkStation],
        raw: Dict[Tuple[ResourceCumulative, WorkStation], int],
        start: Dict[Tuple[TaskUnit, Period], int],
        cyc: Dict[Period, int],
        ramp_up_duration: Optional[float] = None,
        nb_adjustments: Optional[int] = None,
    ):
        super().__init__(problem=problem)
        # wks[t]: workstation assigned to task t
        self.wks = wks
        # raw[(r, w)]: amount of resource r allocated to workstation w
        self.raw = raw
        # start[(t, p)]: start time of task t in period p
        self.start = start
        # cyc[p]: cycle time for period p
        self.cyc = cyc
        # Evaluations
        self.ramp_up_duration = ramp_up_duration
        self.nb_adjustments = nb_adjustments

    def is_allocated(self, task: Task, unary_resource: WorkStation) -> bool:
        return self.wks[task[0]] == unary_resource

    def get_end_time(self, task: Task) -> int:
        return self.get_start_time(task) + self.problem.get_duration(
            task=task[0], p=task[1], w=self.wks[task[0]]
        )

    def get_start_time(self, task: Task) -> int:
        if task in self.start:
            return self.start[task]
        else:
            return 0

    def copy(self) -> "RCALBPLSolution":
        return RCALBPLSolution(
            problem=self.problem,
            wks=dict(self.wks),
            raw=dict(self.raw),
            start=dict(self.start),
            cyc=dict(self.cyc),
            ramp_up_duration=self.ramp_up_duration,
            nb_adjustments=self.nb_adjustments,
        )

    def lazy_copy(self) -> "RCALBPLSolution":
        return RCALBPLSolution(
            problem=self.problem,
            wks=self.wks,
            raw=self.raw,
            start=self.start,
            cyc=self.cyc,
            ramp_up_duration=self.ramp_up_duration,
            nb_adjustments=self.nb_adjustments,
        )

    def change_problem(self, new_problem: Problem) -> None:
        super().change_problem(new_problem=new_problem)
        self.ramp_up_duration = None
        self.nb_adjustments = None


class RCALBPLProblem(Problem):
    """
    Problem definition for Resource-Constrained Assembly Line Balancing
    with Learning Effect (RC-ALBP/L).
    """

    def __init__(
        self,
        c_target: int,
        c_max: int,
        nb_stations: int,
        nb_periods: int,
        nb_tasks: int,
        precedences: List[Tuple[Task, Task]],
        durations: List[List[int]],
        nb_resources: int,
        capa_resources: List[int],
        cons_resources: List[List[int]],
        nb_zones: int,
        capa_zones: List[int],
        cons_zones: List[List[int]],
        neutr_zones: List[List[int]],
        p_start: int = 0,
        p_end: Optional[int] = None,
    ):
        self.c_target = c_target
        self.c_max = c_max
        self.nb_stations = nb_stations
        self.nb_periods = nb_periods
        self.nb_tasks = nb_tasks

        self.precedences = precedences
        self.durations = durations

        self.nb_resources = nb_resources
        self.capa_resources = capa_resources
        self.cons_resources = cons_resources

        self.nb_zones = nb_zones
        self.capa_zones = capa_zones
        self.cons_zones = cons_zones
        self.neutr_zones = neutr_zones

        # Derived properties
        self.tasks = list(range(self.nb_tasks))
        self.stations = list(range(self.nb_stations))
        self.periods = list(range(self.nb_periods))
        self.resources = list(range(self.nb_resources))
        self.zones = list(range(self.nb_zones))

        # Period start/end of interest.
        self.p_start = p_start
        self.p_end = p_end if p_end is not None else self.nb_periods
        self.periods = list(range(self.p_start, self.p_end))  # Restricted to the window

    def get_objective_register(self) -> ObjectiveRegister:
        # Multi-objective optimization: ramp-up duration and line adjustments
        dict_objective = {
            "ramp_up_duration": ObjectiveDoc(
                type=TypeObjective.OBJECTIVE, default_weight=1.0
            ),
            "nb_adjustments": ObjectiveDoc(
                type=TypeObjective.OBJECTIVE, default_weight=1.0
            ),
            # Penalty objectives for constraints violations
            "violation_precedence": ObjectiveDoc(
                type=TypeObjective.PENALTY, default_weight=100000.0
            ),
            "violation_capacity": ObjectiveDoc(
                type=TypeObjective.PENALTY, default_weight=100000.0
            ),
            "violation_cycle_time": ObjectiveDoc(
                type=TypeObjective.PENALTY, default_weight=100000.0
            ),
        }
        return ObjectiveRegister(
            objective_sense=ModeOptim.MINIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc=dict_objective,
        )

    def get_duration(self, task: TaskUnit, p: Period, w: WorkStation) -> int:
        n = max(-1, p - w)
        # durations array has 0 at index 0 (n=-1), so n+1 aligns perfectly
        if n < 0:
            return 0
        return self.durations[task][n]

    def evaluate(self, variable: RCALBPLSolution) -> Dict[str, float]:
        ramp_up_duration = 0
        nb_adjustments = 0

        violation_precedence = 0.0
        violation_capacity = 0.0
        violation_cycle_time = 0.0

        # 1. Ramp-up criteria computation
        costs = {}
        for p in self.periods:
            if p < self.nb_stations:
                costs[p] = variable.cyc[p]  # Unstable period
            else:
                costs[p] = (
                    variable.cyc[p] if variable.cyc[p] > self.c_target else 0
                )  # Stable period
        ramp_up_duration = sum(costs[p] for p in self.periods)
        for p in self.periods:
            if p >= self.nb_stations and (p - 1) in costs:
                if costs[p - 1] != costs[p]:
                    nb_adjustments += 1
        # 2. Evaluate cycle time validity
        for p in self.periods:
            cyc_p = variable.cyc[p]
            if cyc_p < self.c_target or cyc_p > self.c_max:
                print("lower than c-target or upper than cmax")
                violation_cycle_time += 1
            # Unstable periods cannot change their cycle time
            if self.nb_stations > p >= 1:
                if variable.cyc[p] != variable.cyc[p - 1]:
                    print("didn't respect the frozen")
                    violation_cycle_time += 1
            # Check if cycle time is respected by all tasks scheduled in p
            for t in self.tasks:
                w = variable.wks[t]
                dur = self.get_duration(t, p, w)
                end_t = variable.start[(t, p)] + dur
                if end_t > cyc_p:
                    print("Exceeded the cycle at some point")
                    violation_cycle_time += end_t - cyc_p

        # 3. Evaluate Precedence Constraints
        for a, b in self.precedences:
            wa = variable.wks[a]
            wb = variable.wks[b]
            if wa > wb:
                print("Violation station")
                violation_precedence += wa - wb
            # If in the same workstation, check temporal precedence
            if wa == wb:
                for p in self.periods:
                    dur_a = self.get_duration(a, p, wa)
                    end_a = variable.start[(a, p)] + dur_a
                    if end_a > variable.start[(b, p)]:
                        print("Violation precedence, ", end_a, variable.start[(b, p)])
                        violation_precedence += end_a - variable.start[(b, p)]

        # 4. Evaluate Global & Cumulative Resource / Zone Capacities
        for r in self.resources:
            total_r_allocated = sum(variable.raw.get((r, w), 0) for w in self.stations)
            if total_r_allocated > self.capa_resources[r]:  # 0-indexed capacity arrays
                violation_capacity += total_r_allocated - self.capa_resources[r]
                print("too much capa ?")

        # Sweep-line algorithm for cumulative checking per period and workstation
        for p in self.periods:
            for w in self.stations:
                events = []
                tasks_in_w = [t for t in self.tasks if variable.wks[t] == w]
                if not tasks_in_w:
                    continue

                for t in tasks_in_w:
                    s_time = variable.start[(t, p)]
                    dur = self.get_duration(t, p, w)
                    if dur > 0:
                        events.append((s_time, "start", t))
                        events.append((s_time + dur, "end", t))

                events.sort(key=lambda x: (x[0], 0 if x[1] == "end" else 1))

                active_tasks = set()
                for time, ev_type, t in events:
                    if ev_type == "start":
                        active_tasks.add(t)
                        # 4a. Check Cumulative Resources
                        for r in self.resources:
                            usage = sum(
                                self.cons_resources[r][tsk] for tsk in active_tasks
                            )
                            allocated = variable.raw.get((r, w), 0)
                            if usage > allocated:
                                violation_capacity += usage - allocated
                        # 4b. Check Cumulative Zones
                        for z in self.zones:
                            z_usage = sum(
                                self.cons_zones[z][tsk] for tsk in active_tasks
                            )
                            if z_usage > self.capa_zones[z]:
                                violation_capacity += z_usage - self.capa_zones[z - 1]

                        # 4c. Check disabled zones (conflict check) [cite: 285, 291]
                        disabled_zones = set()
                        for tsk in active_tasks:
                            for dz in self.neutr_zones[tsk]:
                                disabled_zones.add(dz)
                        for tsk in active_tasks:
                            for z in self.zones:
                                if self.cons_zones[z][tsk] > 0 and z in disabled_zones:
                                    violation_capacity += 1
                    else:
                        active_tasks.remove(t)
        variable.ramp_up_duration = ramp_up_duration
        variable.nb_adjustments = nb_adjustments
        return {
            "ramp_up_duration": ramp_up_duration,
            "nb_adjustments": nb_adjustments,
            "violation_precedence": violation_precedence,
            "violation_capacity": violation_capacity,
            "violation_cycle_time": violation_cycle_time,
        }

    def compute_actual_cycle_time_per_period(
        self, solution: RCALBPLSolution
    ) -> dict[Period, int]:
        dict_cycle_time_per_period = {}
        for p in self.periods:
            task_this_period = [t for t in solution.start if t[1] == p]
            if len(task_this_period) > 0:
                dict_cycle_time_per_period[p] = max(
                    [solution.get_end_time(t) for t in task_this_period]
                )
            else:
                dict_cycle_time_per_period[p] = None
        return dict_cycle_time_per_period

    def satisfy(self, variable: RCALBPLSolution) -> bool:
        evals = self.evaluate(variable)
        return (
            evals["violation_precedence"] == 0
            and evals["violation_capacity"] == 0
            and evals["violation_cycle_time"] == 0
        )

    def get_dummy_solution(self) -> RCALBPLSolution:
        """
        Creates a trivial dummy solution (likely invalid).
        Assigns all tasks sequentially to the first workstation.
        """
        wks = {t: 1 for t in self.tasks}
        raw = {
            (r, w): self.capa_resources[r] if w == 1 else 0
            for r in self.resources
            for w in self.stations
        }
        cyc = {p: self.c_max for p in self.periods}
        start = {}
        for p in self.periods:
            current_time = 0
            for t in self.tasks:
                start[(t, p)] = current_time
                n = max(-1, p - wks[t])
                dur = self.durations[t][n + 1]
                current_time += dur
        return RCALBPLSolution(self, wks, raw, start, cyc)

    def get_solution_type(self) -> type[Solution]:
        return RCALBPLSolution

    def build_sgs_schedule_for_period_slow(
        self,
        wks: Dict[int, int],
        raw: Dict[Tuple[int, int], int],
        target_starts: Dict[int, int],
        period: int,
    ) -> Tuple[Dict[int, int], int]:
        """
        Robust Serial Generation Scheme (SGS) to compute a feasible schedule.
        Uses a dynamic eligible set to strictly guarantee Precedence constraints,
        and uses 'target_starts' (from an optimal future period) to guide the packing.
        """
        start_times = {}
        end_times = {}
        scheduled_tasks = []

        unscheduled = set(self.tasks)

        while unscheduled:
            # 1. Identify eligible tasks (all same-station predecessors are already scheduled)
            eligible = []
            for t in unscheduled:
                is_eligible = True
                for pred, succ in self.precedences:
                    if succ == t and wks[pred] == wks[t] and pred not in start_times:
                        is_eligible = False
                        break
                if is_eligible:
                    eligible.append(t)

            # 2. Pick the best eligible task guided by the optimal future schedule
            # Primary: Earliest target start time
            # Secondary: Longest Processing Time (LPT) to pack big tasks first
            # Tertiary: Task ID for stable tie-breaking
            t = min(
                eligible,
                key=lambda x: (
                    target_starts.get(x, 0),
                    -self.get_duration(x, period, wks[x]),
                    x,
                ),
            )

            unscheduled.remove(t)
            w = wks[t]
            dur = self.get_duration(t, period, w)

            # 3. Earliest Start Time (EST) based on Precedences
            est = 0
            for pred, succ in self.precedences:
                if succ == t and wks[pred] == w:
                    est = max(est, end_times[pred])

            # 4. Collect candidate times
            candidates = {est}
            for ot in scheduled_tasks:
                if end_times[ot] >= est:
                    candidates.add(end_times[ot])
            candidates = sorted(list(candidates))

            best_start = est
            for c in candidates:
                if dur == 0:
                    best_start = c
                    break

                overlaps = [
                    ot
                    for ot in scheduled_tasks
                    if start_times[ot] < c + dur and end_times[ot] > c
                ]
                if not overlaps:
                    best_start = c
                    break

                events = {c, c + dur}
                for ot in overlaps:
                    events.add(max(c, start_times[ot]))
                    events.add(min(c + dur, end_times[ot]))
                events = sorted(list(events))

                is_valid = True
                for i in range(len(events) - 1):
                    t_eval = events[i]
                    active = [
                        ot
                        for ot in overlaps
                        if start_times[ot] <= t_eval < end_times[ot]
                    ]

                    # A. Global Resources
                    for r in self.resources:
                        if self.cons_resources[r][t] > 0:
                            usage = (
                                sum(self.cons_resources[r][ot] for ot in active)
                                + self.cons_resources[r][t]
                            )
                            if usage > self.capa_resources[r]:
                                is_valid = False
                                break
                    if not is_valid:
                        break

                    # B. Local Resources
                    active_w = [ot for ot in active if wks[ot] == w]
                    for r in self.resources:
                        if self.cons_resources[r][t] > 0:
                            usage = (
                                sum(self.cons_resources[r][ot] for ot in active_w)
                                + self.cons_resources[r][t]
                            )
                            if usage > raw.get((r, w), 0):
                                is_valid = False
                                break
                    if not is_valid:
                        break

                    # C. Zones and Zone Blocking
                    disabled_zones_by_active = {
                        dz for ot in active_w for dz in self.neutr_zones[ot]
                    }
                    disabled_zones_by_t = set(self.neutr_zones[t])

                    for z in self.zones:
                        req_t = self.cons_zones[z][t]
                        req_active = sum(self.cons_zones[z][ot] for ot in active_w)

                        if req_t + req_active > self.capa_zones[z]:
                            is_valid = False
                            break

                        # If t disables z, it cannot overlap with any task that CONSUMES z
                        if z in disabled_zones_by_t and req_active > 0:
                            is_valid = False
                            break

                        # If t consumes z, it cannot overlap with any task that DISABLES z
                        if req_t > 0 and z in disabled_zones_by_active:
                            is_valid = False
                            break

                    if not is_valid:
                        break

                if is_valid:
                    best_start = c
                    break

            start_times[t] = best_start
            end_times[t] = best_start + dur
            scheduled_tasks.append(t)

        cycle_time = max(end_times.values()) if end_times else 0
        return start_times, cycle_time

    def build_sgs_schedule_for_period(
        self,
        wks: Dict[int, int],
        raw: Dict[Tuple[int, int], int],
        target_starts: Dict[int, int],
        period: int,
    ) -> Tuple[Dict[int, int], int]:
        """
        Highly Optimized Serial Generation Scheme (SGS).
        Uses 1D timeline arrays and slice mathematics to evaluate capacities
        in a fraction of a millisecond per task.
        """
        start_times = {}
        end_times = {}

        # --- 1. Precompute Task Data (Avoids dict lookups in tight loops) ---
        t_res = {
            t: [(r, self.cons_resources[r][t]) for r in self.resources]
            for t in self.tasks
        }
        t_res_active = {
            t: [(r, req) for r, req in t_res[t] if req > 0] for t in self.tasks
        }

        t_zones = {
            t: [(z, self.cons_zones[z][t]) for z in self.zones] for t in self.tasks
        }
        t_zones_active = {
            t: [(z, req) for z, req in t_zones[t] if req > 0] for t in self.tasks
        }

        t_disabled_zones = {t: self.neutr_zones[t] for t in self.tasks}

        # --- 2. Initialize Dependency Graph for O(1) Eligibility ---
        local_in_degree = {t: 0 for t in self.tasks}
        local_successors = {t: [] for t in self.tasks}
        est = {t: 0 for t in self.tasks}  # Earliest Start Time tracking

        for pred, succ in self.precedences:
            if wks[pred] == wks[succ]:
                local_in_degree[succ] += 1
                local_successors[pred].append(succ)

        eligible = [t for t in self.tasks if local_in_degree[t] == 0]

        # --- 3. Initialize High-Performance Timeline Arrays ---
        # Instead of sweeping intervals, we maintain the exact usage at each time unit.
        # We start with an array length of c_max. It will expand dynamically if needed.
        current_horizon = self.c_max + 1000

        res_usage = [[0] * current_horizon for _ in self.resources]
        res_w_usage = {
            w: [[0] * current_horizon for _ in self.resources] for w in self.stations
        }
        zone_usage = {
            w: [[0] * current_horizon for _ in self.zones] for w in self.stations
        }
        zone_disabled = {
            w: [[0] * current_horizon for _ in self.zones] for w in self.stations
        }

        end_events = {0}  # Track interesting candidate start times

        # --- 4. Main Scheduling Loop ---
        for _ in range(self.nb_tasks):
            # Pick the best eligible task
            t = min(
                eligible,
                key=lambda x: (
                    target_starts.get(x, 0),
                    -self.get_duration(x, period, wks[x]),  # LPT tie-breaker
                    x,
                ),
            )
            eligible.remove(t)

            w = wks[t]
            dur = self.get_duration(t, period, w)

            # Find candidate times strictly >= Earliest Start Time
            candidates = sorted([c for c in end_events if c >= est[t]])

            best_start = est[t]

            for c in candidates:
                if dur == 0:
                    best_start = c
                    break

                end = c + dur

                # Dynamically expand arrays if the schedule exceeds the expected horizon
                if end >= current_horizon:
                    extend_len = max(1000, end - current_horizon + 1000)
                    for r in self.resources:
                        res_usage[r].extend([0] * extend_len)
                        for w_idx in self.stations:
                            res_w_usage[w_idx][r].extend([0] * extend_len)
                    for z in self.zones:
                        for w_idx in self.stations:
                            zone_usage[w_idx][z].extend([0] * extend_len)
                            zone_disabled[w_idx][z].extend([0] * extend_len)
                    current_horizon += extend_len

                valid = True

                # A. Fast Global & Local Resource Check
                for r, req in t_res_active[t]:
                    # The max() over a slice is evaluated in C and is lightning fast
                    if max(res_usage[r][c:end]) + req > self.capa_resources[r]:
                        valid = False
                        break
                    if max(res_w_usage[w][r][c:end]) + req > raw.get((r, w), 0):
                        valid = False
                        break
                if not valid:
                    continue

                # B. Fast Zone Capacity & Zone Blocking Check
                for z, req in t_zones_active[t]:
                    if max(zone_usage[w][z][c:end]) + req > self.capa_zones[z]:
                        valid = False
                        break
                    # RULE 2: If consuming, cannot overlap with disabled
                    if max(zone_disabled[w][z][c:end]) > 0:
                        valid = False
                        break
                if not valid:
                    continue

                # RULE 1: If disabling, cannot overlap with any consumption
                for dz in t_disabled_zones[t]:
                    if max(zone_usage[w][dz][c:end]) > 0:
                        valid = False
                        break
                if not valid:
                    continue

                # If we pass all checks, this time slot is mathematically perfect!
                best_start = c
                break

            # --- 5. Commit the Task ---
            start_times[t] = best_start
            end_t = best_start + dur
            end_times[t] = end_t
            end_events.add(end_t)

            # Update the timelines for the committed interval
            if dur > 0:
                for r, req in t_res_active[t]:
                    for time_idx in range(best_start, end_t):
                        res_usage[r][time_idx] += req
                        res_w_usage[w][r][time_idx] += req
                for z, req in t_zones_active[t]:
                    for time_idx in range(best_start, end_t):
                        zone_usage[w][z][time_idx] += req
                for dz in t_disabled_zones[t]:
                    for time_idx in range(best_start, end_t):
                        zone_disabled[w][dz][time_idx] += 1

            # --- 6. Update Dependencies ---
            for succ in local_successors[t]:
                local_in_degree[succ] -= 1
                est[succ] = max(est[succ], end_t)
                if local_in_degree[succ] == 0:
                    eligible.append(succ)

        cycle_time = max(end_times.values()) if end_times else 0
        return start_times, cycle_time


def parse_rcalbpl_json(file_path: str) -> RCALBPLProblem:
    """
    Parses the RC-ALBP/L JSON data and constructs the Problem instance.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    # 1. Base Variables
    c_target = data["c_target"]
    c_max = data["c_max"]
    nb_stations = data["nb_stations"]
    nb_periods = data["nb_periods"]
    nb_tasks = data["nb_tasks"]

    # 2. Precedences
    precedences = [(p[0] - 1, p[1] - 1) for p in data.get("precedences", [])]

    # 3. Durations matrix
    durations = data.get("durations", [])

    # 4. Resources
    nb_resources = data.get("nb_resources", 0)
    capa_resources = data.get("capa_resources", [])
    cons_resources = data.get("cons_resources", [])

    # 5. Zones
    nb_zones = data.get("nb_zones", 0)
    capa_zones = data.get("capa_zones", [])
    cons_zones = data.get("cons_zones", [])
    neutr_zones = data.get("neutr_zones", [])
    neutr_zones = [[i - 1 for i in x] for x in neutr_zones]

    return RCALBPLProblem(
        c_target=c_target,
        c_max=c_max,
        nb_stations=nb_stations,
        nb_periods=nb_periods,
        nb_tasks=nb_tasks,
        precedences=precedences,
        durations=durations,
        nb_resources=nb_resources,
        capa_resources=capa_resources,
        cons_resources=cons_resources,
        nb_zones=nb_zones,
        capa_zones=capa_zones,
        cons_zones=cons_zones,
        neutr_zones=neutr_zones,
    )
