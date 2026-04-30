"""
CP-SAT Solvers for RC-ALBP with Shared Resources

This module shows how each modeling approach handles shared resources:
- FOLDED: Works naturally ✓
- CALENDAR: Works naturally ✓
- UNFOLDED: BREAKS! The time-unfolding trick doesn't work for shared resources ✗

WHY UNFOLDED BREAKS:
In unfolded time, tasks on different stations exist in different time windows.
A shared resource constraint needs to consider tasks across ALL time windows
simultaneously, which defeats the purpose of the unfolding.
"""

from enum import Enum
from typing import Any

from ortools.sat.python.cp_model import CpSolverSolutionCallback

from discrete_optimization.alb.rcalbp.problem import (
    RCALBPProblem,
    RCALBPSolution,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
)
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver


class ModelingShared(Enum):
    FOLDED = 0
    CALENDAR = 1


class CpSatRcAlbpSolver(OrtoolsCpSatSolver):
    """
    CP-SAT Solver for RC-ALBP with shared resources.
    Implements FOLDED and CALENDAR approaches.
    """

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
        task_station = {}
        task_station_binary = {}
        for task in self.problem.tasks:
            for station in self.problem.stations:
                task_station_binary[(task, station)] = self.cp_model.NewBoolVar(
                    name=f"task_{task}_{station}"
                )
            self.cp_model.add_exactly_one(
                [
                    task_station_binary[(task, station)]
                    for station in self.problem.stations
                ]
            )
            task_station[task] = sum(
                [
                    station_to_idx[s] * task_station_binary[(task, s)]
                    for s in self.problem.stations
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
        self.variables["task_station"] = task_station
        self.variables["idx_to_station"] = idx_to_station
        self.variables["makespan"] = makespan

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


def demonstrate_shared_resources():
    """Demonstrate solving with shared resources."""
    print("\n" + "=" * 80)
    print("  RC-ALBP WITH SHARED RESOURCES")
    print("=" * 80 + "\n")

    # Test on simple example
    print("=" * 80)
    print("TEST 1: Small Example with Shared AGV Resource")
    print("=" * 80 + "\n")

    for modeling in [ModelingShared.FOLDED, ModelingShared.CALENDAR]:
        print(f"\n{modeling.name} Model:")
        print("-" * 60)

        solver = CpSatRcAlbpSolver(problem)
        params_cp = ParametersCp.default_cpsat()

        result_storage = solver.solve(
            modeling=modeling,
            parameters_cp=params_cp,
            time_limit=30,
            ortools_cpsat_solver_kwargs={"log_search_progress": False},
        )

        if len(result_storage) > 0:
            solution = result_storage.get_best_solution()
            eval_result = problem.evaluate(solution)

            print(f"  Cycle Time: {solution.cycle_time}")
            print(f"  Valid (all constraints): {problem.satisfy(solution)}")
            print(
                f"  Station resource penalty: {eval_result.get('penalty_resource_station', 0)}"
            )
            print(
                f"  Shared resource penalty: {eval_result.get('penalty_resource_shared', 0)}"
            )
            print(f"  Task assignments:")
            for station in problem.stations:
                tasks_on_station = [
                    t for t in problem.tasks if solution.task_assignment[t] == station
                ]
                print(f"    {station}: {tasks_on_station}")
        else:
            print("  No solution found")

    # Explain why UNFOLDED doesn't work
    print("\n" + "=" * 80)
    print("  WHY UNFOLDED MODEL DOESN'T WORK FOR SHARED RESOURCES")
    print("=" * 80 + "\n")

    print("In the UNFOLDED model, time is partitioned by station:")
    print("  Station 0: time ∈ [0, cycle_time)")
    print("  Station 1: time ∈ [cycle_time, 2*cycle_time)")
    print("  Station 2: time ∈ [2*cycle_time, 3*cycle_time)")
    print()
    print("For a SHARED resource constraint to work, we need to check that at")
    print("any GLOBAL time t, the sum of demands across ALL stations ≤ capacity.")
    print()
    print("But in unfolded time:")
    print("  - A task at t=5 on Station 0 exists at global time 5")
    print("  - A task at t=5 on Station 1 exists at global time cycle_time + 5")
    print("  - These are DIFFERENT time points!")
    print()
    print("The unfolding trick works for station-specific resources because")
    print("each station's time window is independent. But shared resources")
    print("require reasoning about ALL stations simultaneously, which breaks")
    print("the independence assumption.")
    print()
    print("✓ FOLDED: Works (uses optional intervals)")
    print("✓ CALENDAR: Works (uses global cumulative for shared resources)")
    print("✗ UNFOLDED: Cannot model shared resources naturally")

    print("\n" + "=" * 80)
    print("  DEMONSTRATION COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    demonstrate_shared_resources()
