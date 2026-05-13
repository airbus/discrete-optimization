#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""
CP-SAT Solvers for RC-ALBP with Shared Resources

This module shows how each modeling approach handles shared resources:
- FOLDED: Works naturally ✓
- CALENDAR: Works naturally ✓
"""

from enum import Enum
from typing import Any

from ortools.sat.python.cp_model import (
    CpSolverSolutionCallback,
    IntervalVar,
    LinearExprT,
)

from discrete_optimization.alb.rcalbp.problem import (
    RCALBPProblem,
    RCALBPSolution,
    Resource,
    Task,
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.solvers.cpsat.allocation import (
    AllocationCpSatSolver,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
)


class ModelingShared(Enum):
    FOLDED = 0
    CALENDAR = 1


class CpSatRcAlbpSolver(
    AllocationCpSatSolver[Task, UnaryResource],
):
    """
    CP-SAT Solver for RC-ALBP with shared resources.
    Implements FOLDED and CALENDAR approaches.
    """

    # def get_binary_allocation_variable(self, task: Task, unary_resource: UnaryResource) -> LinearExprT:
    #    if self.modeling == ModelingShared.FOLDED:
    #        station_to_idx = {s: i for i, s in enumerate(self.problem.stations)}
    #        return self.variables["task_station_binary"][task, station_to_idx[unary_resource]]
    #    raise NotImplementedError

    # def get_integer_allocation_variable(self, task: Task) -> LinearExprT:
    #    if self.modeling == ModelingShared.FOLDED:
    #        return self.variables["task_station"][task]
    #    raise NotImplementedError

    def get_resource_consumption_intervals(
        self, resource: Resource
    ) -> list[tuple[IntervalVar, int]]:
        if self.modeling == ModelingShared.FOLDED:
            return [
                (
                    self.variables["intervals"][t],
                    self.problem.get_task_demand(t, resource),
                )
                for t in self.problem.tasks
            ]

    def get_task_start_or_end_variable(
        self, task: Task, start_or_end: StartOrEnd
    ) -> LinearExprT:
        if start_or_end == StartOrEnd.START:
            return self.variables["starts"][task]
        return self.variables["ends"][task]

    def get_task_unary_resource_is_present_variable(
        self, task: Task, unary_resource: UnaryResource
    ) -> LinearExprT:
        if self.modeling == ModelingShared.FOLDED:
            return self.variables["task_station_binary"][task, unary_resource]
        raise NotImplementedError

    hyperparameters = [
        EnumHyperparameter(
            enum=ModelingShared, name="modeling", default=ModelingShared.FOLDED
        )
    ]
    problem: RCALBPProblem

    def __init__(self, problem: RCALBPProblem):
        super().__init__(problem)
        self.variables = {}
        self.modeling: ModelingShared = None

    def init_model(self, **kwargs: Any) -> None:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        self.modeling = kwargs["modeling"]
        if self.modeling == ModelingShared.FOLDED:
            self.init_model_folded(**kwargs)
        elif self.modeling == ModelingShared.CALENDAR:
            self.init_model_calendar(**kwargs)

    def init_model_folded(self, **kwargs):
        """
        FOLDED model with shared resources.
        """
        super().init_model(**kwargs)
        horizon = sum(self.problem.task_times.values())
        station_to_idx = {s: i for i, s in enumerate(self.problem.stations)}
        idx_to_station = {i: s for s, i in station_to_idx.items()}
        # ------------------------------------------------------------------
        # 1. Task Assignment Variables
        # ------------------------------------------------------------------
        task_station_binary = self.create_allocation()
        task_station = {
            task: sum(
                [
                    station_to_idx[s] * task_station_binary[(task, s)]
                    for s in self.problem.stations
                ]
            )
            for task in self.problem.tasks
        }
        for task in self.problem.tasks:
            self.cp_model.add_exactly_one(
                [
                    task_station_binary[(task, station)]
                    for station in self.problem.stations
                ]
            )

        # ------------------------------------------------------------------
        # 2. Start Time and Interval Variables
        # ------------------------------------------------------------------
        starts = {}
        ends = {}
        intervals = {}

        for task in self.problem.tasks:
            duration = self.problem.task_times[task]
            start = self.cp_model.NewIntVar(0, horizon, f"start_{task}")
            end = self.cp_model.NewIntVar(0, horizon, f"end_{task}")
            interval = self.cp_model.NewIntervalVar(
                start, duration, end, f"interval_{task}"
            )
            starts[task] = start
            ends[task] = end
            intervals[task] = interval

        # ------------------------------------------------------------------
        # 3. Station Precedence Constraints
        # ------------------------------------------------------------------
        for pred, succ in self.problem.precedences:
            self.cp_model.Add(task_station[pred] <= task_station[succ])

        # ------------------------------------------------------------------
        # 4. Temporal Precedence (Same Station)
        # ------------------------------------------------------------------
        for pred, succ in self.problem.precedences:
            same_station = self.cp_model.NewBoolVar(f"same_{pred}_{succ}")
            self.cp_model.Add(task_station[pred] == task_station[succ]).OnlyEnforceIf(
                same_station
            )
            self.cp_model.Add(task_station[pred] != task_station[succ]).OnlyEnforceIf(
                same_station.Not()
            )
            self.cp_model.Add(ends[pred] <= starts[succ]).OnlyEnforceIf(same_station)

        # ------------------------------------------------------------------
        # 5. STATION-SPECIFIC Resource Constraints (Optional Intervals)
        # ------------------------------------------------------------------
        optional_intervals = {}
        station_specific_resources = self.problem.get_station_specific_resources()

        for task in self.problem.tasks:
            for station_idx, station in enumerate(self.problem.stations):
                duration = self.problem.task_times[task]
                opt_interval = self.cp_model.NewOptionalIntervalVar(
                    starts[task],
                    duration,
                    ends[task],
                    task_station_binary[(task, station)],
                    f"opt_interval_{task}_{station}",
                )
                optional_intervals[(task, station_idx)] = opt_interval

        # Cumulative constraints for station-specific resources
        for station_idx, station in enumerate(self.problem.stations):
            for resource in station_specific_resources:
                station_intervals = []
                station_demands = []
                for task in self.problem.tasks:
                    demand = self.problem.get_task_demand(task, resource)
                    if demand > 0:
                        station_intervals.append(
                            optional_intervals[(task, station_idx)]
                        )
                        station_demands.append(demand)

                capacity = self.problem.get_station_capacity(station, resource)
                if station_intervals and capacity > 0:
                    self.cp_model.AddCumulative(
                        station_intervals, station_demands, capacity
                    )

        # ------------------------------------------------------------------
        # 6. SHARED Resource Constraints (Global Cumulative)
        # ------------------------------------------------------------------
        # This is the key difference: shared resources use ALL task intervals,
        # not just those assigned to a specific station

        for resource in self.problem.shared_resources:
            global_intervals = []
            global_demands = []

            for task in self.problem.tasks:
                demand = self.problem.get_task_demand(task, resource)
                if demand > 0:
                    global_intervals.append(intervals[task])
                    global_demands.append(demand)

            capacity = self.problem.shared_resource_capacities.get(resource, 0)
            if global_intervals and capacity > 0:
                self.cp_model.AddCumulative(global_intervals, global_demands, capacity)

        # ------------------------------------------------------------------
        # 7. Objective: Minimize Cycle Time
        # ------------------------------------------------------------------
        makespan = self.cp_model.NewIntVar(0, horizon, "makespan")
        self.cp_model.AddMaxEquality(makespan, list(ends.values()))
        self.cp_model.Minimize(makespan)

        # Store variables
        self.variables["starts"] = starts
        self.variables["ends"] = ends
        self.variables["intervals"] = intervals
        self.variables["opt_intervals"] = optional_intervals
        self.variables["task_station_binary"] = task_station_binary
        self.variables["task_station"] = task_station
        self.variables["idx_to_station"] = idx_to_station
        self.variables["makespan"] = makespan

    def create_allocation(self):
        tasks, unary_resources = self.get_default_tasks_n_unary_resources()
        task_to_station = {}
        for task in tasks:
            for unary_resource in unary_resources:
                if self.problem.is_compatible_task_unary_resource(
                    task=task, unary_resource=unary_resource
                ):
                    boolvar = self.cp_model.new_bool_var(
                        f"is_present_{task}_{unary_resource}"
                    )
                    task_to_station[(task, unary_resource)] = boolvar
        return task_to_station

    def init_model_calendar(self, **kwargs):
        """
        CALENDAR model - optimized based on shared resource presence.

        Two paths:
        1. NO shared resources: Simple calendar model with absolute time
        2. WITH shared resources: Complex model with cycle-time decomposition

        Key insight for shared resources:
        We need TWO sets of intervals:
        1. Absolute time intervals [start, end) for NoOverlap and station assignment
        2. Cycle time intervals [cycle_start, cycle_end) for shared resources

        Where: cycle_start = start % cycle_time
        """
        super().init_model(**kwargs)

        horizon = sum(self.problem.task_times.values())
        station_to_idx = {s: i for i, s in enumerate(self.problem.stations)}
        idx_to_station = {i: s for s, i in station_to_idx.items()}

        cycle_time = self.cp_model.NewIntVar(lb=1, ub=horizon, name="cycle_time")
        nb_stations = len(self.problem.stations)

        # ------------------------------------------------------------------
        # 1. Task Variables (Absolute Time)
        # ------------------------------------------------------------------
        starts = {}
        ends = {}
        intervals = {}

        for task in self.problem.tasks:
            duration = self.problem.task_times[task]
            start = self.cp_model.NewIntVar(0, horizon, f"start_{task}")
            end = self.cp_model.NewIntVar(0, horizon, f"end_{task}")
            interval = self.cp_model.NewIntervalVar(
                start, duration, end, f"interval_{task}"
            )
            starts[task] = start
            ends[task] = end
            intervals[task] = interval

        # ------------------------------------------------------------------
        # 2. STATION-SPECIFIC Resources: Calendar Tasks
        # ------------------------------------------------------------------
        station_specific_resources = self.problem.get_station_specific_resources()

        # Compute max capacity per station-specific resource
        max_capacity_station = {}
        for resource in station_specific_resources:
            max_cap = 0
            for station in self.problem.stations:
                cap = self.problem.get_station_capacity(station, resource)
                max_cap = max(max_cap, cap)
            max_capacity_station[resource] = max_cap

        # Create calendar tasks for station-specific resources
        calendar_intervals = {}
        calendar_demands = {}

        for station_idx, station in enumerate(self.problem.stations):
            for resource in station_specific_resources:
                station_cap = self.problem.get_station_capacity(station, resource)
                max_cap = max_capacity_station[resource]
                blocking_demand = max_cap - station_cap

                if blocking_demand > 0:
                    cal_interval = self.cp_model.NewIntervalVar(
                        station_idx * cycle_time,
                        cycle_time,
                        (station_idx + 1) * cycle_time,
                        f"cal_interval_{station}_{resource}",
                    )
                    calendar_intervals[(station, resource)] = cal_interval
                    calendar_demands[(station, resource)] = blocking_demand

        # Add cumulative constraints for station-specific resources
        for resource in station_specific_resources:
            max_cap = max_capacity_station[resource]
            if max_cap == 0:
                continue

            resource_intervals = []
            resource_demands = []

            # Real task intervals
            for task in self.problem.tasks:
                demand = self.problem.get_task_demand(task, resource)
                if demand > 0:
                    resource_intervals.append(intervals[task])
                    resource_demands.append(demand)

            # Calendar blocking intervals
            for (station, res), cal_interval in calendar_intervals.items():
                if res == resource:
                    resource_intervals.append(cal_interval)
                    resource_demands.append(calendar_demands[(station, res)])

            if resource_intervals:
                self.cp_model.AddCumulative(
                    resource_intervals, resource_demands, max_cap
                )

        # ------------------------------------------------------------------
        # 3. SHARED Resources: Cycle Time Intervals
        # ------------------------------------------------------------------
        # OPTIMIZATION: Only create cycle-time decomposition if there are shared resources
        # This saves variables and constraints when not needed

        has_shared_resources = bool(self.problem.shared_resources)

        if has_shared_resources:
            # Complex path: Create cycle-time decomposition for shared resources
            station_vars = {}
            cycle_starts = {}
            cycle_ends = {}
            cycle_intervals = {}

            for task in self.problem.tasks:
                duration = self.problem.task_times[task]

                # Station assignment variable (implicit in absolute time)
                station_var = self.cp_model.NewIntVar(
                    0, nb_stations - 1, f"station_{task}"
                )
                station_vars[task] = station_var

                # Cycle time coordinates: cycle_start = start % cycle_time
                cycle_start = self.cp_model.NewIntVar(0, horizon, f"cycle_start_{task}")
                cycle_end = self.cp_model.NewIntVar(0, horizon, f"cycle_end_{task}")

                # Bounds: cycle_start ∈ [0, cycle_time), cycle_end ≤ cycle_time
                self.cp_model.Add(cycle_start >= 0)
                self.cp_model.Add(cycle_start < cycle_time)
                self.cp_model.Add(cycle_end <= cycle_time)

                # Link: start = station * cycle_time + cycle_start
                station_offset = self.cp_model.NewIntVar(0, horizon, f"offset_{task}")
                self.cp_model.AddMultiplicationEquality(
                    station_offset, [station_var, cycle_time]
                )

                # Enforce: start = station_offset + cycle_start
                self.cp_model.Add(starts[task] == station_offset + cycle_start)

                # Enforce: end = station_offset + cycle_end
                self.cp_model.Add(ends[task] == station_offset + cycle_end)

                # Interval duration constraint
                self.cp_model.Add(cycle_end == cycle_start + duration)

                # Create cycle interval for shared resources
                cycle_interval = self.cp_model.NewIntervalVar(
                    cycle_start, duration, cycle_end, f"cycle_interval_{task}"
                )

                cycle_starts[task] = cycle_start
                cycle_ends[task] = cycle_end
                cycle_intervals[task] = cycle_interval

            # Add cumulative constraints on CYCLE intervals for shared resources
            for resource in self.problem.shared_resources:
                global_intervals = []
                global_demands = []

                for task in self.problem.tasks:
                    demand = self.problem.get_task_demand(task, resource)
                    if demand > 0:
                        global_intervals.append(cycle_intervals[task])
                        global_demands.append(demand)

                capacity = self.problem.shared_resource_capacities.get(resource, 0)
                if global_intervals and capacity > 0:
                    self.cp_model.AddCumulative(
                        global_intervals, global_demands, capacity
                    )

        # ------------------------------------------------------------------
        # 4. NoOverlap Constraint (Prevent Spanning Boundaries)
        # ------------------------------------------------------------------
        # Create dummy intervals at station boundaries
        dummy_intervals = []
        for station_idx in range(len(self.problem.stations)):
            dummy_interval = self.cp_model.NewIntervalVar(
                station_idx * cycle_time,
                0,
                station_idx * cycle_time,
                f"dummy_interval_{station_idx}",
            )
            dummy_intervals.append(dummy_interval)

        # For EACH task, add NoOverlap with dummy intervals
        # This prevents that task from spanning station boundaries
        for t in intervals.values():
            self.cp_model.add_no_overlap([t] + dummy_intervals)

        # ------------------------------------------------------------------
        # 5. Precedence Constraints
        # ------------------------------------------------------------------
        for pred, succ in self.problem.precedences:
            self.cp_model.Add(ends[pred] <= starts[succ])

        # ------------------------------------------------------------------
        # 6. Objective: Minimize Cycle Time
        # ------------------------------------------------------------------
        for task in self.problem.tasks:
            self.cp_model.Add(ends[task] <= len(self.problem.stations) * cycle_time)

        self.cp_model.Minimize(cycle_time)

        # Store variables
        self.variables["starts"] = starts
        self.variables["ends"] = ends
        self.variables["cycle_time"] = cycle_time
        self.variables["idx_to_station"] = idx_to_station
        self.variables["has_shared_resources"] = has_shared_resources

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> RCALBPSolution:
        """Extract solution from solver."""
        if self.modeling == ModelingShared.FOLDED:
            starts = self.variables["starts"]
            task_station = self.variables["task_station"]
            idx_to_station = self.variables["idx_to_station"]

            task_assignment = {}
            task_schedule = {}

            for task in self.problem.tasks:
                station_idx = cpsolvercb.Value(task_station[task])
                station = idx_to_station[station_idx]
                task_assignment[task] = station

                start_time = cpsolvercb.Value(starts[task])
                task_schedule[task] = start_time

            cycle_time = int(cpsolvercb.ObjectiveValue())

            return RCALBPSolution(
                problem=self.problem,
                task_assignment=task_assignment,
                task_schedule=task_schedule,
                cycle_time=cycle_time,
            )

        else:  # ModelingShared.CALENDAR
            cycle_time = cpsolvercb.Value(self.variables["cycle_time"])
            task_assignment = {}
            task_schedule = {}

            for task in self.problem.tasks:
                start = cpsolvercb.Value(self.variables["starts"][task])
                end = cpsolvercb.Value(self.variables["ends"][task])
                station_idx = start // cycle_time

                if start == end and station_idx >= len(self.problem.stations):
                    station_idx = len(self.problem.stations) - 1
                    start_modulo = cycle_time
                else:
                    start_modulo = start % cycle_time

                task_assignment[task] = self.problem.stations[station_idx]
                task_schedule[task] = start_modulo

            return RCALBPSolution(
                problem=self.problem,
                task_assignment=task_assignment,
                task_schedule=task_schedule,
                cycle_time=cycle_time,
            )
