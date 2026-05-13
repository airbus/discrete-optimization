#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""
Base class for Assembly Line Balancing Problems.

Provides common functionality using standard mixins:
- PrecedenceProblem: handles precedence constraints and graph
- AllocationProblem: handles task-to-station assignment
- SchedulingProblem: handles timing and makespan
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import Dict, Generic, List, Optional, Tuple, TypeVar

import networkx as nx

from discrete_optimization.generic_tasks_tools.allocation import (
    AllocationProblem,
    AllocationSolution,
)
from discrete_optimization.generic_tasks_tools.precedence import PrecedenceProblem
from discrete_optimization.generic_tasks_tools.scheduling import (
    SchedulingProblem,
    SchedulingSolution,
)
from discrete_optimization.generic_tools.do_problem import Solution

# Type variables for generic ALB problems
Task = TypeVar("Task", bound=Hashable)
Station = TypeVar("Station", bound=Hashable)  # Stations are unary resources


class TaskData:
    """
    Base class for task data in ALB problems.

    This class is designed to be subclassed to add domain-specific attributes
    like resource requirements, zone constraints, etc.

    Attributes:
        task_id: Unique identifier for the task
        processing_time: Duration to execute the task (in base conditions)
    """

    def __init__(self, task_id: Hashable, processing_time: int):
        """
        Initialize task data.

        Args:
            task_id: Unique identifier for the task
            processing_time: Duration to execute the task
        """
        self.task_id = task_id
        self.processing_time = processing_time

    def __hash__(self):
        return hash(self.task_id)

    def __eq__(self, other):
        if isinstance(other, TaskData):
            return self.task_id == other.task_id
        return False

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(id={self.task_id}, time={self.processing_time})"
        )


class ResourceTaskData(TaskData):
    """
    Task data with resource requirements for resource-constrained ALB problems.

    Extends TaskData with resource consumption information.

    Attributes:
        task_id: Unique identifier for the task
        processing_time: Duration to execute the task
        resource_consumption: Dict mapping resource_name -> consumption amount
    """

    def __init__(
        self,
        task_id: Hashable,
        processing_time: int,
        resource_consumption: Dict[Hashable, int] = None,
    ):
        """
        Initialize resource task data.

        Args:
            task_id: Unique identifier for the task
            processing_time: Duration to execute the task
            resource_consumption: Dict mapping resource_name -> consumption (default: {})
        """
        super().__init__(task_id, processing_time)
        self.resource_consumption = (
            resource_consumption if resource_consumption is not None else {}
        )

    def get_resource_consumption(self, resource: Hashable) -> int:
        """
        Get resource consumption for a given resource.

        Args:
            resource: Resource identifier

        Returns:
            Consumption amount (0 if not specified)
        """
        return self.resource_consumption.get(resource, 0)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(id={self.task_id}, "
            f"time={self.processing_time}, "
            f"resources={self.resource_consumption})"
        )


class BaseALBSolution(
    SchedulingSolution[Task],
    AllocationSolution[Task, Station],
    Generic[Task, Station],
):
    """
    Base solution class for Assembly Line Balancing problems.

    Provides time-related helper methods that work for all ALB variants:
    - get_start_time_in_cycle(task): Start time within the cycle
    - get_absolute_start_time(task): "Unfolded" absolute time
    - get_station_index(task): Index of assigned station

    Subclasses must implement:
    - task_assignment: Dict[Task, Station] mapping
    - Either task_schedule OR compute schedule on demand
    - cycle_time attribute
    """

    problem: "BaseALBProblem[Task, Station]"

    def get_station_index(self, task: Task) -> int:
        """
        Get the index of the station where task is assigned.

        Returns:
            Station index (0-based)
        """
        raise NotImplementedError("Subclass must implement get_station_index()")

    def get_start_time_in_cycle(self, task: Task) -> int:
        """
        Get the start time of a task within its cycle.

        For problems with explicit scheduling (RC-ALBP), this returns task_schedule[task].
        For problems without scheduling (SALBP), this computes it using greedy scheduling.

        Returns:
            Start time within the cycle [0, cycle_time)
        """
        raise NotImplementedError("Subclass must implement get_start_time_in_cycle()")

    def get_absolute_start_time(self, task: Task) -> int:
        """
        Get the "unfolded" absolute start time of a task.

        This is useful for precedence checking and visualization.
        Formula: station_index * cycle_time + start_time_in_cycle

        Returns:
            Absolute start time in unfolded timeline
        """
        station_idx = self.get_station_index(task)
        start_in_cycle = self.get_start_time_in_cycle(task)
        cycle_time = self._get_cycle_time()
        return station_idx * cycle_time + start_in_cycle

    def get_end_time_in_cycle(self, task: Task) -> int:
        """
        Get the end time of a task within its cycle.

        Returns:
            End time within the cycle
        """
        start_in_cycle = self.get_start_time_in_cycle(task)
        duration = self.problem.task_times[task]
        return start_in_cycle + duration

    def get_absolute_end_time(self, task: Task) -> int:
        """
        Get the "unfolded" absolute end time of a task.

        Returns:
            Absolute end time in unfolded timeline
        """
        abs_start = self.get_absolute_start_time(task)
        duration = self.problem.task_times[task]
        return abs_start + duration

    def _get_cycle_time(self) -> int:
        """
        Get the cycle time.

        Priority:
        1. Explicit cycle_time attribute on solution (if set)
        2. Compute from schedule: max end_time_in_cycle across all tasks
        3. Problem's cycle_time (for problems with fixed cycle time like SALBP)

        Returns:
            Cycle time value
        """
        # 1. Explicit cycle_time on solution
        if hasattr(self, "cycle_time") and self.cycle_time is not None:
            return self.cycle_time

        # 2. Compute from schedule (for problems where cycle_time is decision variable)
        max_end_time = max(
            self.get_end_time_in_cycle(task) for task in self.problem.tasks
        )
        if max_end_time > 0:
            return max_end_time

        # 3. Fall back to problem's cycle_time
        if hasattr(self.problem, "cycle_time"):
            return self.problem.cycle_time

        raise ValueError("Cycle time not available")

    def _compute_greedy_schedule(
        self, tasks_per_station: Optional[Dict[Station, List[Task]]] = None
    ) -> Dict[Task, int]:
        """
        Compute greedy schedule from task assignments.

        This is a template method that should be overridden by subclasses
        to implement problem-specific scheduling logic:
        - SALBP: Sequential execution (no resources)
        - RC-ALBP: Resource-constrained serial generation scheme

        Args:
            tasks_per_station: Optional pre-computed station grouping

        Returns:
            Dict mapping task -> start_time_in_cycle

        Raises:
            NotImplementedError: Subclasses must implement this
        """
        raise NotImplementedError(
            "Subclass must implement _compute_greedy_schedule() with "
            "problem-specific scheduling logic"
        )

    def _get_task_station(self, task: Task) -> Station:
        """Get station assignment for a task (must be implemented by subclass)."""
        raise NotImplementedError("Subclass must implement _get_task_station()")


class BaseALBProblem(
    PrecedenceProblem[Task],
    AllocationProblem[Task, Station],
    SchedulingProblem[Task],
    Generic[Task, Station],
):
    """
    Base class for all Assembly Line Balancing Problems.

    Combines three standard mixins:
    - PrecedenceProblem: handles precedence constraints and graph
    - AllocationProblem: handles task-to-station assignment
    - SchedulingProblem: handles timing and makespan

    All ALB variants extend this base with additional constraints
    (cycle time, resources, learning effects, etc.).

    Attributes:
        tasks_data: List of TaskData objects (or subclass) for each task
        task_times: Dict mapping task_id -> processing_time (for convenience)
        precedences: List of (predecessor, successor) pairs
        tasks: Sorted list of task identifiers
        stations: List of workstation identifiers
        nb_tasks: Number of tasks
        nb_stations: Number of stations
    """

    def __init__(
        self,
        tasks_data: List[TaskData],
        precedences: List[Tuple[Task, Task]],
        stations: List[Station],
    ):
        """
        Initialize base ALB problem.

        Args:
            tasks_data: List of TaskData objects (or subclass) containing task info
            precedences: List of (predecessor_id, successor_id) pairs
            stations: List of workstation identifiers
        """
        # Store task data objects
        self.tasks_data = list(tasks_data)

        # Build convenience dictionary for quick lookup
        self.task_times = {td.task_id: td.processing_time for td in self.tasks_data}
        self.task_data_by_id = {td.task_id: td for td in self.tasks_data}

        # Precedences and stations
        self.precedences = precedences
        self.stations = list(stations)

        # Task list (sorted for determinism)
        self.tasks = sorted([td.task_id for td in self.tasks_data], key=str)

        # Derived attributes
        self.nb_tasks = len(self.tasks)
        self.nb_stations = len(self.stations)

        # Build precedence structures for PrecedenceProblem mixin
        self._build_precedence_structures()

    def get_task_data(self, task_id: Task) -> TaskData:
        """
        Get TaskData object for a given task.

        Args:
            task_id: Task identifier

        Returns:
            TaskData object (or subclass) for this task
        """
        return self.task_data_by_id[task_id]

    def _build_precedence_structures(self):
        """Build successors and predecessors dicts."""
        self._successors_dict = {t: [] for t in self.tasks}
        self._predecessors_dict = {t: [] for t in self.tasks}

        for pred, succ in self.precedences:
            if pred in self._successors_dict:
                self._successors_dict[pred].append(succ)
            if succ in self._predecessors_dict:
                self._predecessors_dict[succ].append(pred)

    # === PrecedenceProblem interface ===

    @property
    def tasks_list(self) -> List[Task]:
        """Return the list of all tasks."""
        return self.tasks

    def get_precedence_constraints(self) -> Dict[Task, Iterable[Task]]:
        """
        Map each task to its successors.

        Required by PrecedenceProblem mixin.

        Returns:
            Dictionary mapping task -> list of successor tasks
        """
        return self._successors_dict

    def _topological_sort_station_tasks(self) -> List[Task]:
        """
        Topological sort of tasks respecting precedence constraints.

        Args:
            tasks: List of tasks to sort

        Returns:
            Topologically sorted list
        """
        graph = self.get_precedence_graph()
        nx_graph = graph.to_networkx()
        order = list(nx.topological_sort(nx_graph))
        return order

    # === AllocationProblem interface ===

    @property
    def unary_resources_list(self) -> List[Station]:
        """
        Return the list of all stations (unary resources).

        In ALB problems, stations are the unary resources for allocation.
        """
        return self.stations

    # === SchedulingProblem interface ===

    def get_makespan_upper_bound(self) -> int:
        """
        Return an upper bound on the makespan.

        Worst case: all tasks executed sequentially.

        Returns:
            Sum of all task processing times
        """
        return sum(self.task_times.values())

    # === ALB-specific helper methods ===

    def get_predecessors(self) -> Dict[Task, List[Task]]:
        """
        Get predecessor mapping (inverse of successors).

        Returns:
            Dictionary mapping task -> list of predecessor tasks
        """
        return self._predecessors_dict

    def get_successors(self) -> Dict[Task, List[Task]]:
        """
        Get successor mapping.

        Returns:
            Dictionary mapping task -> list of successor tasks
        """
        return self._successors_dict

    def get_last_tasks(self) -> List[Task]:
        """
        Get tasks with no successors (sink nodes).

        Returns:
            List of tasks that are not predecessors to any other task
        """
        return [t for t in self.tasks if len(self._successors_dict[t]) == 0]

    def get_first_tasks(self) -> List[Task]:
        """
        Get tasks with no predecessors (source nodes).

        Returns:
            List of tasks that have no predecessor constraints
        """
        return [t for t in self.tasks if len(self._predecessors_dict[t]) == 0]

    def check_precedence_violation_stations(
        self, pred_station: Station, succ_station: Station
    ) -> bool:
        """
        Check if assigning predecessor to pred_station and successor to
        succ_station violates precedence at the station level.

        In ALB, precedence means predecessor must be assigned to an earlier
        or equal station (by index in stations list).

        Args:
            pred_station: Station assigned to predecessor task
            succ_station: Station assigned to successor task

        Returns:
            True if this assignment violates precedence, False otherwise
        """
        try:
            pred_idx = self.stations.index(pred_station)
            succ_idx = self.stations.index(succ_station)
            return pred_idx > succ_idx  # Violation if pred comes after succ
        except ValueError:
            # Station not in list - consider it a violation
            return True

    # === Evaluation and Satisfaction (Base Implementation) ===

    def evaluate_base_constraints(
        self, solution: "BaseALBSolution[Task, Station]"
    ) -> Dict[str, float]:
        """
        Evaluate base constraints common to all ALB problems.

        Uses the new solution time helper methods for clean precedence checking.

        Returns:
            Dictionary with:
            - penalty_precedence: Number of precedence violations
            - penalty_unscheduled: Number of tasks without assignment or schedule

        Subclasses should call this and add their specific penalties.
        """
        penalty_precedence = 0.0

        # 1. Check if all tasks are assigned
        assigned_tasks = set()
        for task in self.tasks:
            try:
                # Access station assignment
                station = solution._get_task_station(task)
                if station is not None:
                    assigned_tasks.add(task)
            except (KeyError, NotImplementedError, AttributeError):
                pass

        penalty_unscheduled = float(self.nb_tasks - len(assigned_tasks))

        # 2. Check precedence constraints using absolute times
        # This is much cleaner than the old approach!
        for pred, succ in self.precedences:
            # Skip if tasks aren't assigned
            if pred not in assigned_tasks or succ not in assigned_tasks:
                penalty_precedence += 1
                continue

            try:
                # Use absolute times for global precedence check
                pred_end = solution.get_absolute_end_time(pred)
                succ_start = solution.get_absolute_start_time(succ)

                if pred_end > succ_start:
                    penalty_precedence += 1

            except (KeyError, NotImplementedError, AttributeError):
                # If we can't get times, assume violation
                penalty_precedence += 1

        return {
            "penalty_precedence": penalty_precedence,
            "penalty_unscheduled": penalty_unscheduled,
        }

    def evaluate(self, solution: Solution) -> Dict[str, float]:
        """
        Default evaluate implementation using base constraints.

        Subclasses should override this to add problem-specific objectives
        and constraints. They can call super().evaluate() or use
        evaluate_base_constraints() to get common penalties.

        Returns:
            Dictionary with evaluation metrics
        """
        return self.evaluate_base_constraints(solution)

    def satisfy(self, solution: Solution) -> bool:
        """
        Base satisfaction check: all penalties must be zero.

        Subclasses should override evaluate() to add specific constraints.

        Args:
            solution: Solution to check

        Returns:
            True if solution satisfies all constraints
        """
        eval_dict = self.evaluate(solution)

        # Check all penalty fields (any key starting with 'penalty_')
        for key, value in eval_dict.items():
            if key.startswith("penalty_") and value > 0:
                return False

        return True

    def correct_allocation_to_station_precedence(
        self, allocation_to_station: list[int]
    ) -> bool:
        topological = self._topological_sort_station_tasks()
        index_to_id = {i: self.tasks_data[i].task_id for i in range(self.nb_tasks)}
        id_to_index = {index_to_id[x]: x for x in index_to_id}
        for task in topological:
            index_task = id_to_index[task]
            for pred in self.get_predecessors()[task]:
                index_pred = id_to_index[pred]
                allocation_to_station[index_task] = max(
                    allocation_to_station[index_pred], allocation_to_station[index_task]
                )
        return allocation_to_station
