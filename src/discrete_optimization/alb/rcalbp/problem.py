"""
Resource-Constrained Assembly Line Balancing Problem (RC-ALBP)

This problem combines elements from:
- SALBP (Simple Assembly Line Balancing): Task assignment and precedence
- Resource constraints at workstations

Problem Definition:
-------------------
Given:
- A fixed number of workstations (stations)
- A set of tasks with processing times, precedence constraints, and resource requirements
- Multiple resource types with two allocation models:

  1. STATION-SPECIFIC RESOURCES:
     - Shared taxonomy/names across all stations (e.g., "Workbench", "Robot_Arm")
     - Tasks consume resources by name (e.g., Task_A needs 1 "Workbench")
     - Each station has its own CAPACITY for each resource type
     - Example: Station_1 has 2 Workbenches, Station_2 has 1 Workbench
     - A task can only execute at a station with sufficient resource capacity

  2. SHARED/GLOBAL RESOURCES (optional):
     - Global pool shared across ALL stations (e.g., "AGV", "Quality_Inspector")
     - Tasks consume from the global capacity pool
     - Total concurrent demand across all stations must not exceed global capacity

Decision Variables:
- Assignment of tasks to workstations
- Start time of each task within the cycle

Objective:
- Minimize the cycle time (makespan)

Constraints:
1. Precedence: Predecessor tasks must finish before successors (or be assigned to earlier stations)
2. Station-specific Resource Capacity: At each station, cumulative resource consumption cannot
   exceed that station's capacity for each resource type (tasks CAN overlap if total demand fits)
3. Shared Resource Capacity: Across ALL stations simultaneously, cumulative resource consumption
   cannot exceed global capacity for each shared resource
"""

from copy import deepcopy
from typing import Dict, Hashable, List, Optional, Set, Tuple

import numpy as np

from discrete_optimization.alb.base.problem import (
    BaseALBProblem,
    BaseALBSolution,
    ResourceTaskData,
)
from discrete_optimization.generic_tasks_tools.allocation import (
    UnaryResource,
)
from discrete_optimization.generic_tools.do_problem import (
    EncodingRegister,
    ModeOptim,
    ObjectiveDoc,
    ObjectiveHandling,
    ObjectiveRegister,
    Problem,
    Solution,
    TypeObjective,
)
from discrete_optimization.generic_tools.encoding_register import ListInteger

# Type aliases for clarity
Task = Hashable
Station = Hashable
Resource = Hashable


class RCALBPSolution(BaseALBSolution[Task, Station]):
    """
    Solution for RC-ALBP problem.

    RC-ALBP has explicit scheduling (task_schedule) since resource constraints
    require knowing exact start times within the cycle.

    Uses vectorized representation for task assignments (like SALBP):
    - allocation_to_station[i] = station index for task i
    - task_assignment property builds dict on-demand

    Attributes:
        problem: Reference to the problem instance
        allocation_to_station: List[int] where index i = task i, value = station index
        task_schedule: Dict mapping each task to its start time within cycle
        cycle_time: The computed cycle time (max end time across all stations)
    """

    problem: "RCALBPProblem"

    def __init__(
        self,
        problem: "RCALBPProblem",
        allocation_to_station: Optional[List[int]] = None,
        task_assignment: Optional[Dict[Task, Station]] = None,
        task_schedule: Optional[Dict[Task, int]] = None,
        cycle_time: Optional[int] = None,
    ):
        """
        Initialize RC-ALBP solution.

        Args:
            problem: Problem instance
            allocation_to_station: Vectorized task assignments [task_idx -> station_idx]
            task_assignment: Dict task assignments (legacy, converted to vector)
            task_schedule: Optional schedule (dict)
            cycle_time: Optional cycle time

        Note: Provide either allocation_to_station OR task_assignment, not both.
        """
        super().__init__(problem)

        # Vectorized representation (preferred)
        if allocation_to_station is not None:
            self.allocation_to_station = allocation_to_station
        elif task_assignment is not None:
            # Convert dict to vector for backward compatibility
            self.allocation_to_station = [0] * problem.nb_tasks
            for task, station in task_assignment.items():
                task_idx = problem.tasks_to_index[task]
                station_idx = problem.stations.index(station)
                self.allocation_to_station[task_idx] = station_idx
        else:
            # Empty solution
            self.allocation_to_station = [0] * problem.nb_tasks

        self.task_schedule = task_schedule if task_schedule is not None else {}
        self.cycle_time = cycle_time
        self._cached_schedule = None  # Cache for greedy schedule
        self._cached_task_assignment = None  # Cache for dict conversion

    @property
    def task_assignment(self) -> Dict[Task, Station]:
        """
        Build task_assignment dict from vectorized representation on-demand.

        This allows backward compatibility with code expecting dict.
        """
        if self._cached_task_assignment is None:
            self._cached_task_assignment = {}
            for i, station_idx in enumerate(self.allocation_to_station):
                task = self.problem.tasks[i]
                station = self.problem.stations[station_idx]
                self._cached_task_assignment[task] = station
        return self._cached_task_assignment

    # BaseALBSolution interface implementation
    def get_station_index(self, task: Task) -> int:
        """Get the index of the station where task is assigned."""
        task_idx = self.problem.tasks_to_index[task]
        return self.allocation_to_station[task_idx]

    def get_start_time_in_cycle(self, task: Task) -> int:
        """
        Get start time within cycle.

        If explicit schedule exists, use it.
        Otherwise, compute greedy schedule from task assignments.
        """
        # If we have explicit schedule, use it
        if self.task_schedule:
            return self.task_schedule.get(task, 0)

        # Otherwise, compute greedy schedule using serial generation
        if self._cached_schedule is None:
            self._cached_schedule = self._compute_greedy_schedule()

        return self._cached_schedule.get(task, 0)

    def _get_task_station(self, task: Task) -> Station:
        """Get station assignment for a task."""
        task_idx = self.problem.tasks_to_index[task]
        station_idx = self.allocation_to_station[task_idx]
        return self.problem.stations[station_idx]

    def _compute_greedy_schedule(
        self, tasks_per_station: Optional[Dict[Station, List[Task]]] = None
    ) -> Dict[Task, int]:
        """
        Compute resource-feasible schedule using Serial Generation Scheme (SGS).

        Given task-to-station assignments, builds a schedule that:
        - Respects precedence constraints
        - Respects station-specific resource capacities
        - Respects shared resource capacities

        This is a greedy decoder/repair heuristic that finds earliest feasible
        start times for each task.

        Args:
            tasks_per_station: Optional pre-computed station grouping

        Returns:
            Dict mapping task -> start_time_in_cycle
        """
        if tasks_per_station is None:
            # Group tasks by station
            tasks_per_station = {}
            for task in self.problem.tasks:
                station = self._get_task_station(task)
                if station not in tasks_per_station:
                    tasks_per_station[station] = []
                tasks_per_station[station].append(task)

        schedule = {}

        # Track which tasks have been scheduled
        scheduled = set()

        # Build predecessor count for each task
        pred_count = {}
        for task in self.problem.tasks:
            preds = self.problem.get_predecessors().get(task, [])
            pred_count[task] = len(preds)

        # Compute a reasonable time horizon
        total_duration = sum(self.problem.task_times.values())
        time_horizon = total_duration * 2  # Upper bound

        # Initialize resource usage profiles
        # Station-specific: {station: {resource: [usage_over_time]}}
        station_resource_usage = {}
        for station in self.problem.stations:
            station_resource_usage[station] = {}
            for resource in self.problem.get_station_specific_resources():
                station_resource_usage[station][resource] = [0] * time_horizon

        # Shared resources: {resource: [usage_over_time]}
        shared_resource_usage = {}
        for resource in self.problem.shared_resources:
            shared_resource_usage[resource] = [0] * time_horizon

        # Helper: Check if resources are available at [start, end)
        def resources_available(
            task: Task, station: Station, start: int, end: int
        ) -> bool:
            """Check if task can execute at [start, end) without violating resources."""
            if end > time_horizon:
                return False

            # Check station-specific resources
            for resource in self.problem.get_station_specific_resources():
                demand = self.problem.get_task_demand(task, resource)
                if demand > 0:
                    capacity = self.problem.get_station_capacity(station, resource)
                    for t in range(start, end):
                        if (
                            station_resource_usage[station][resource][t] + demand
                            > capacity
                        ):
                            return False

            # Check shared resources
            for resource in self.problem.shared_resources:
                demand = self.problem.get_task_demand(task, resource)
                if demand > 0:
                    capacity = self.problem.get_shared_capacity(resource)
                    for t in range(start, end):
                        if shared_resource_usage[resource][t] + demand > capacity:
                            return False

            return True

        # Helper: Update resource usage when task is scheduled
        def reserve_resources(task: Task, station: Station, start: int, end: int):
            """Reserve resources for task executing at [start, end)."""
            # Station-specific resources
            for resource in self.problem.get_station_specific_resources():
                demand = self.problem.get_task_demand(task, resource)
                if demand > 0:
                    for t in range(start, end):
                        station_resource_usage[station][resource][t] += demand

            # Shared resources
            for resource in self.problem.shared_resources:
                demand = self.problem.get_task_demand(task, resource)
                if demand > 0:
                    for t in range(start, end):
                        shared_resource_usage[resource][t] += demand

        # Serial Generation Scheme: schedule tasks one by one
        while len(scheduled) < len(self.problem.tasks):
            # Find eligible tasks (all predecessors scheduled)
            eligible = []
            for task in self.problem.tasks:
                if task in scheduled:
                    continue
                if pred_count[task] == 0:
                    eligible.append(task)

            if not eligible:
                # Deadlock - shouldn't happen with valid precedence
                # Schedule remaining tasks anyway
                eligible = [t for t in self.problem.tasks if t not in scheduled]

            # Sort eligible tasks for determinism (e.g., by task ID)
            eligible.sort(key=str)

            # Schedule ONE task per iteration (first eligible that fits)
            scheduled_this_round = False
            for task in eligible:
                station = self._get_task_station(task)
                duration = self.problem.task_times[task]

                # Compute earliest start time based on predecessors
                min_start_precedence = 0
                for pred in self.problem.get_predecessors().get(task, []):
                    if pred in schedule:
                        pred_end = schedule[pred] + self.problem.task_times[pred]
                        min_start_precedence = max(min_start_precedence, pred_end)

                # Find earliest time slot where resources are available
                start_time = min_start_precedence
                while start_time < time_horizon:
                    end_time = start_time + duration

                    if resources_available(task, station, start_time, end_time):
                        # Schedule the task
                        schedule[task] = start_time
                        reserve_resources(task, station, start_time, end_time)
                        scheduled.add(task)
                        scheduled_this_round = True

                        # Update predecessor counts for successors
                        for succ in self.problem.get_successors().get(task, []):
                            pred_count[succ] -= 1

                        break

                    # Try next time slot
                    start_time += 1

                if task in scheduled:
                    break  # Schedule one task at a time

            if not scheduled_this_round and eligible:
                # Force schedule first eligible task even if resources tight
                task = eligible[0]
                station = self._get_task_station(task)
                duration = self.problem.task_times[task]

                min_start = 0
                for pred in self.problem.get_predecessors().get(task, []):
                    if pred in schedule:
                        min_start = max(
                            min_start, schedule[pred] + self.problem.task_times[pred]
                        )

                schedule[task] = min_start
                scheduled.add(task)

                # Update predecessor counts
                for succ in self.problem.get_successors().get(task, []):
                    pred_count[succ] -= 1

        return schedule

    # Legacy SchedulingSolution interface (for backward compatibility)
    def get_start_time(self, task: Task) -> int:
        """Return the start time of a task within cycle."""
        return self.get_start_time_in_cycle(task)

    def get_end_time(self, task: Task) -> int:
        """Return the end time of a task within cycle."""
        return self.get_end_time_in_cycle(task)

    # AllocationSolution interface
    def is_allocated(self, task: Task, unary_resource: UnaryResource) -> bool:
        """Check if a task is allocated to a given station."""
        task_idx = self.problem.tasks_to_index[task]
        station_idx = self.allocation_to_station[task_idx]
        return self.problem.stations[station_idx] == unary_resource

    def copy(self) -> "RCALBPSolution":
        """Create a deep copy of the solution."""
        return RCALBPSolution(
            problem=self.problem,
            allocation_to_station=deepcopy(self.allocation_to_station),
            task_schedule=deepcopy(self.task_schedule),
        )

    def change_problem(self, new_problem: Problem) -> Solution:
        """Change the problem instance."""
        return RCALBPSolution(
            problem=new_problem,
            allocation_to_station=deepcopy(self.allocation_to_station),
            task_schedule=deepcopy(self.task_schedule),
            cycle_time=self.cycle_time,
        )

    def __str__(self):
        return f"RCALBPSolution(cycle_time={self.cycle_time})"

    def __hash__(self):
        return hash(
            (
                tuple(self.allocation_to_station),
                tuple(sorted(self.task_schedule.items())),
            )
        )

    def __eq__(self, other):
        return (
            self.allocation_to_station == other.allocation_to_station
            and self.task_schedule == other.task_schedule
        )


class RCALBPProblem(BaseALBProblem[Task, Station]):
    """
    Resource-Constrained Assembly Line Balancing Problem.

    Extends BaseALBProblem with cumulative resource constraints.
    Supports both station-specific and shared/global resources.

    Inherits from:
    - BaseALBProblem: Provides precedence, allocation, and scheduling foundations

    Attributes (in addition to BaseALBProblem):
        resources: List of station-specific resource names (shared taxonomy)
        station_resources: Capacity per station {station: {resource: capacity}}
        shared_resources: Set of shared resource names (global pool)
        shared_resource_capacities: Global capacity {resource: capacity}
        nb_resources: Total count (station-specific + shared)
    """

    # Resource accessor methods (object-oriented API)
    def get_task_demand(self, task: Task, resource: Resource) -> int:
        """
        Get resource demand for a task (0 if not specified).

        Uses ResourceTaskData if available, otherwise falls back to empty demand.

        Args:
            task: Task identifier
            resource: Resource identifier

        Returns:
            Consumption amount (0 if not specified)
        """
        task_data = self.get_task_data(task)
        if isinstance(task_data, ResourceTaskData):
            return task_data.get_resource_consumption(resource)
        return 0

    def get_station_capacity(self, station: Station, resource: Resource) -> int:
        """Get resource capacity at a station (0 if not specified)."""
        return self.station_resources.get(station, {}).get(resource, 0)

    def get_shared_capacity(self, resource: Resource) -> int:
        """Get global capacity for a shared resource (0 if not specified)."""
        return self.shared_resource_capacities.get(resource, 0)

    def is_resource_shared(self, resource: Resource) -> bool:
        """Check if a resource is shared (global pool) or station-specific."""
        return resource in self.shared_resources

    def get_all_resources(self) -> Set[Resource]:
        """Get all resources (station-specific + shared)."""
        return set(self.resources) | self.shared_resources

    def get_station_specific_resources(self) -> Set[Resource]:
        """Get only station-specific resources."""
        return set(self.resources)

    def __init__(
        self,
        tasks_data: List[ResourceTaskData],
        precedences: List[Tuple[Task, Task]],
        stations: List[Station],
        resources: List[Resource] = None,
        station_resources: Dict[Station, Dict[Resource, int]] = None,
        shared_resources: Set[Resource] = None,
        shared_resource_capacities: Dict[Resource, int] = None,
    ):
        """
        Initialize RC-ALBP problem using ResourceTaskData.

        Args:
            tasks_data: List of ResourceTaskData objects (with resource requirements)
            precedences: List of (predecessor, successor) pairs
            stations: List of station identifiers (can be str or int)
            resources: List of station-specific resource names (shared taxonomy)
            station_resources: Capacity per station {station: {resource: capacity}}
            shared_resources: Set of shared resource names (global pool)
            shared_resource_capacities: Global capacity {resource: capacity}

        Example:
            # Create tasks with resource requirements
            tasks_data = [
                ResourceTaskData(task_id="T1", processing_time=5,
                                resource_consumption={"Workbench": 1}),
                ResourceTaskData(task_id="T2", processing_time=3,
                                resource_consumption={"Robot": 1, "AGV": 1}),
            ]

            # Station-specific resources (different capacity per station)
            station_resources = {
                "S1": {"Workbench": 2, "Robot": 1},
                "S2": {"Workbench": 1, "Robot": 2},
            }

            # Shared resources (global pool)
            shared_resources = {"AGV"}
            shared_resource_capacities = {"AGV": 2}
        """
        # Initialize base ALB problem
        super().__init__(
            tasks_data=tasks_data,
            precedences=precedences,
            stations=stations,
        )

        # Station-specific resources
        self.resources = list(resources) if resources is not None else []
        self.station_resources = (
            station_resources if station_resources is not None else {}
        )

        # Shared/global resources
        self.shared_resources = (
            shared_resources if shared_resources is not None else set()
        )
        self.shared_resource_capacities = (
            shared_resource_capacities if shared_resource_capacities is not None else {}
        )

        # Validation: resources should not overlap
        resources_set = set(self.resources)
        overlap = resources_set & self.shared_resources
        if overlap:
            raise ValueError(
                f"Resources cannot be both station-specific and shared: {overlap}\n"
                f"Station-specific resources go in 'resources' list.\n"
                f"Shared resources go in 'shared_resources' set."
            )

        # Resource count
        self.nb_resources = len(self.resources) + len(self.shared_resources)

        # Task index mapping for vectorized solution encoding (like SALBP)
        self.tasks_to_index = {self.tasks[i]: i for i in range(len(self.tasks))}

        # Backward compatibility: keep successors/predecessors as aliases
        self.successors = self.get_successors()
        self.predecessors = self.get_predecessors()

    def get_objective_register(self) -> ObjectiveRegister:
        """
        Define objectives and penalties.

        Primary objective: Minimize cycle time
        Penalties: Precedence violations, resource capacity violations (station + shared)
        """
        return ObjectiveRegister(
            objective_sense=ModeOptim.MINIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc={
                "cycle_time": ObjectiveDoc(
                    type=TypeObjective.OBJECTIVE, default_weight=1.0
                ),
                "penalty_precedence": ObjectiveDoc(
                    type=TypeObjective.PENALTY, default_weight=10000.0
                ),
                "penalty_resource_station": ObjectiveDoc(
                    type=TypeObjective.PENALTY, default_weight=10000.0
                ),
                "penalty_resource_shared": ObjectiveDoc(
                    type=TypeObjective.PENALTY, default_weight=10000.0
                ),
                "penalty_unscheduled": ObjectiveDoc(
                    type=TypeObjective.PENALTY, default_weight=10000.0
                ),
            },
        )

    def evaluate(self, solution: RCALBPSolution) -> Dict[str, float]:
        """
        Evaluate the solution quality and constraint satisfaction.

        Uses base class for precedence checking (via absolute times) and adds
        RC-ALBP-specific resource constraint validation.

        Returns:
            Dictionary with:
            - cycle_time: Maximum makespan across all stations
            - penalty_precedence: Number of precedence violations (from base)
            - penalty_resource_station: Amount of station-specific resource over-consumption
            - penalty_resource_shared: Amount of shared resource over-consumption
        """
        # Get base penalties (precedence + unscheduled) using clean absolute time checking
        base_eval = self.evaluate_base_constraints(solution)

        # Early exit if tasks aren't properly scheduled
        if base_eval["penalty_unscheduled"] > 0:
            return {
                "cycle_time": 0.0,
                "penalty_resource_station": 0.0,
                "penalty_resource_shared": 0.0,
                **base_eval,
            }

        # RC-ALBP-specific: Compute cycle time and validate resources
        station_makespans = {s: 0 for s in self.stations}

        for task in solution.task_assignment:
            end = solution.get_end_time(task)
            station = solution.task_assignment[task]
            station_makespans[station] = max(station_makespans.get(station, 0), end)

        cycle_time = max(station_makespans.values()) if station_makespans else 0
        solution.cycle_time = cycle_time

        # Check resource constraints using numpy arrays (efficient!)
        penalty_resource_station = 0.0
        penalty_resource_shared = 0.0

        if cycle_time > 0:
            # Station-specific resources
            penalty_resource_station = self._validate_station_resources_numpy(
                solution, cycle_time
            )

            # Shared resources
            penalty_resource_shared = self._validate_shared_resources_numpy(
                solution, cycle_time
            )

        return {
            "cycle_time": float(cycle_time),
            "penalty_resource_station": penalty_resource_station,
            "penalty_resource_shared": penalty_resource_shared,
            **base_eval,  # Add penalty_precedence and penalty_unscheduled
        }

    def _validate_station_resources_numpy(
        self, solution: RCALBPSolution, cycle_time: int
    ) -> float:
        """
        Validate station-specific resource constraints using numpy arrays.

        Returns:
            Total penalty (sum of violations across all stations and resources)
        """
        penalty = 0.0

        for station in self.stations:
            station_tasks = [
                t for t, s in solution.task_assignment.items() if s == station
            ]
            if not station_tasks:
                continue

            # Create time-indexed resource usage array
            time_horizon = int(np.ceil(cycle_time)) + 1

            for resource in self.resources:
                capacity = self.get_station_capacity(station, resource)
                if capacity == 0:
                    continue

                # Build resource usage profile over time
                usage = np.zeros(time_horizon, dtype=np.int32)

                for task in station_tasks:
                    demand = self.get_task_demand(task, resource)
                    if demand > 0:
                        start = solution.get_start_time_in_cycle(task)
                        end = start + self.task_times[task]
                        # Mark resource usage during [start, end)
                        usage[start:end] += demand

                # Check for violations
                violations = usage - capacity
                violations = violations[violations > 0]  # Keep only positive violations
                if len(violations) > 0:
                    penalty += np.sum(violations)

        return float(penalty)

    def _validate_shared_resources_numpy(
        self, solution: RCALBPSolution, cycle_time: int
    ) -> float:
        """
        Validate shared resource constraints using numpy arrays.

        For shared resources, we check global usage across ALL stations.

        Returns:
            Total penalty (sum of violations across all shared resources)
        """
        penalty = 0.0

        if not self.shared_resources:
            return penalty

        time_horizon = int(np.ceil(cycle_time)) + 1

        for resource in self.shared_resources:
            capacity = self.get_shared_capacity(resource)
            if capacity == 0:
                continue

            # Build GLOBAL resource usage profile (across all stations)
            usage = np.zeros(time_horizon, dtype=np.int32)

            for task in solution.task_assignment:
                demand = self.get_task_demand(task, resource)
                if demand > 0:
                    start = solution.get_start_time_in_cycle(task)
                    end = start + self.task_times[task]
                    # Mark resource usage during [start, end)
                    usage[start:end] += demand

            # Check for violations
            violations = usage - capacity
            violations = violations[violations > 0]
            if len(violations) > 0:
                penalty += np.sum(violations)

        return float(penalty)

    # satisfy() is inherited from BaseALBProblem - checks all penalty_* == 0

    def get_attribute_register(self) -> EncodingRegister:
        """
        Register for vectorized encoding (enables GA/metaheuristics).

        Solution is encoded as a list of integers (station assignments).
        allocation_to_station[i] = station index for task i
        """
        encoding = {}
        encoding["allocation_to_station"] = ListInteger(
            length=self.nb_tasks, lows=0, ups=self.nb_stations - 1
        )
        return EncodingRegister(encoding)

    def get_solution_type(self) -> type[Solution]:
        """Return the solution class type."""
        return RCALBPSolution

    def get_dummy_solution(self) -> RCALBPSolution:
        """
        Create a trivial dummy solution (likely infeasible).
        Assigns all tasks to the first station sequentially.
        """
        # Vectorized: all tasks to station 0
        allocation_to_station = [0] * self.nb_tasks
        return RCALBPSolution(
            problem=self,
            allocation_to_station=allocation_to_station,
        )
