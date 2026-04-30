#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import math
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

from discrete_optimization.alb.base.problem import (
    BaseALBProblem,
    BaseALBSolution,
    TaskData,
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
from discrete_optimization.generic_tools.graph_api import Graph

Task = int
Resource = int


class SalbpSolution(BaseALBSolution[Task, Resource]):
    """
    Solution for SALBP problem.

    SALBP only assigns tasks to stations without explicit scheduling.
    Scheduling is computed on-demand using greedy sequential scheduling.
    """

    problem: "SalbpProblem"

    def __init__(self, problem: "SalbpProblem", allocation_to_station: list[int]):
        super().__init__(problem)
        self.task_alloc = []
        self.allocation_to_station = allocation_to_station
        self._nb_stations = len(set(self.allocation_to_station))
        self._cached_schedule = None  # Cache for greedy schedule

    # BaseALBSolution interface implementation
    def get_station_index(self, task: Task) -> int:
        """Get the index of the station where task is assigned."""
        return self.allocation_to_station[self.problem.tasks_to_index[task]]

    def get_start_time_in_cycle(self, task: Task) -> int:
        """
        Get start time within cycle using greedy scheduling.

        Since SALBP doesn't have explicit scheduling, we compute it on-demand
        and cache the result.
        """
        if self._cached_schedule is None:
            self._cached_schedule = self._compute_greedy_schedule()
        return self._cached_schedule.get(task, 0)

    def _get_task_station(self, task: Task) -> Resource:
        """Get station assignment for a task."""
        return self.allocation_to_station[self.problem.tasks_to_index[task]]

    def _compute_greedy_schedule(
        self, tasks_per_station: Optional[Dict[int, List[Task]]] = None
    ) -> Dict[Task, int]:
        """
        Compute greedy schedule for SALBP.

        SALBP-specific: Tasks at the same station run SEQUENTIALLY (no parallelism).
        This is different from RC-ALBP where tasks can overlap if resources allow.

        The base class implementation allows parallelism by considering predecessor
        end times. For SALBP, we want strictly sequential execution: each task
        starts immediately when the previous task ends.

        Args:
            tasks_per_station: Optional pre-computed station grouping

        Returns:
            Dict mapping task -> start_time_in_cycle
        """
        if tasks_per_station is None:
            # Group tasks by station
            tasks_per_station = {}
            for i, station in enumerate(self.allocation_to_station):
                if station not in tasks_per_station:
                    tasks_per_station[station] = []
                tasks_per_station[station].append(self.problem.tasks[i])

        schedule = {}
        topo_sort = self.problem._topological_sort_station_tasks()
        for station, tasks in tasks_per_station.items():
            if not tasks:
                continue
            # Topological sort of tasks at this station
            sorted_tasks = sorted(tasks, key=lambda x: topo_sort.index(x))
            # SEQUENTIAL scheduling: each task starts when previous ends
            # No parallelism allowed in SALBP!
            current_time = 0
            for task in sorted_tasks:
                schedule[task] = current_time
                current_time += self.problem.task_times[task]

        return schedule

    def get_end_time(self, task: Task) -> int:
        return self.allocation_to_station[self.problem.tasks_to_index[task]] + 1

    def get_start_time(self, task: Task) -> int:
        return self.allocation_to_station[self.problem.tasks_to_index[task]]

    def is_allocated(self, task: Task, unary_resource: UnaryResource) -> bool:
        return (
            self.allocation_to_station[self.problem.tasks_to_index[task]]
            == unary_resource
        )

    def copy(self) -> "SalbpSolution":
        return SalbpSolution(self.problem, deepcopy(self.allocation_to_station))

    def change_problem(self, new_problem: "Problem") -> "Solution":
        return SalbpSolution(new_problem, deepcopy(self.allocation_to_station))

    def __str__(self):
        return f"SalbpSolution(nb_stations={self._nb_stations})"

    def __hash__(self):
        return hash(tuple(self.allocation_to_station))

    def __eq__(self, other):
        return self.allocation_to_station == other.allocation_to_station


class SalbpProblem(BaseALBProblem[int, int]):
    """
    Simple Assembly Line Balancing Problem.

    Extends BaseALBProblem with cycle time constraints.
    Uses PrecedenceProblem mixin for precedence graph handling.

    Attributes:
        cycle_time: Maximum allowed time per station
        number_of_tasks: Total number of tasks (for compatibility)
        tasks_to_index: Mapping from task_id to index
    """

    def __init__(
        self,
        tasks_data: List[TaskData],
        cycle_time: int,
        precedences: List[Tuple[int, int]],
        number_of_stations: int = None,
    ):
        """
        Initialize SALBP problem.

        Args:
            tasks_data: List of TaskData objects for each task
            cycle_time: Maximum allowed time per station
            precedences: List of (predecessor, successor) pairs
            number_of_stations: Number of available stations (default: nb_tasks)
        """
        # Determine number of stations (upper bound = number of tasks)
        if number_of_stations is None:
            number_of_stations = len(tasks_data)

        # Create stations list
        stations = list(range(number_of_stations))

        # Initialize base class
        super().__init__(
            tasks_data=tasks_data,
            precedences=precedences,
            stations=stations,
        )

        # SALBP-specific attributes
        self.cycle_time = cycle_time

        # Backward compatibility: keep adj and predecessors as aliases
        self.adj = self.get_successors()
        self.predecessors = self.get_predecessors()
        self.precedence = precedences  # For backward compat

        # Task index mapping for list-based solution encoding
        self.tasks_to_index = {self.tasks[i]: i for i in range(len(self.tasks))}

    def evaluate(self, variable: SalbpSolution) -> Dict[str, float]:
        """
        Evaluate the solution.

        Objective: Minimize the number of stations.
        Constraints:
        - Cycle time: Each station's workload must not exceed cycle_time
        - Precedence: Handled by base class using absolute times

        Returns:
            Dictionary with:
            - nb_stations: Number of used stations (objective)
            - penalty_overtime: Sum of time exceeding cycle_time per station
            - penalty_undertime: Sum of idle time per station
            - penalty_precedence: Number of precedence violations (from base)
        """
        # Get base penalties (precedence + unscheduled)
        base_eval = self.evaluate_base_constraints(variable)

        # SALBP-specific: Group by station and check cycle time
        stations_time_cumul = {}
        for i in range(self.nb_tasks):
            station_task_i = variable.allocation_to_station[i]
            if station_task_i not in stations_time_cumul:
                stations_time_cumul[station_task_i] = 0
            stations_time_cumul[station_task_i] += self.task_times[self.tasks_list[i]]

        penalty_overtime = 0.0
        penalty_undertime = 0.0
        for s_id, total_time in stations_time_cumul.items():
            penalty_overtime += max(0, total_time - self.cycle_time)
            penalty_undertime += max(0, self.cycle_time - total_time)

        nb_stations = len(stations_time_cumul)

        # Combine base and SALBP-specific results
        return {
            "nb_stations": nb_stations,
            "penalty_overtime": penalty_overtime,
            "penalty_undertime": penalty_undertime,
            **base_eval,  # Add penalty_precedence and penalty_unscheduled
        }

    def get_solution_type(self) -> type[Solution]:
        return SalbpSolution

    def satisfy(self, variable: SalbpSolution) -> bool:
        """
        Strict check for validity.
        """
        eval_res = self.evaluate(variable)
        return eval_res["penalty_overtime"] == 0 and eval_res["penalty_precedence"] == 0

    def get_attribute_register(self) -> EncodingRegister:
        """
        Register for standard encoding (optional, but good for GA/CP solvers).
        We define the solution as a list of integers (station assignments).
        """
        # Note: Bounding the number of stations is tricky without a heuristic.
        # A safe upper bound is the number of tasks (1 task per station).
        encoding = dict()
        encoding[f"allocation_to_station"] = ListInteger(
            length=self.nb_tasks, lows=0, ups=self.nb_tasks - 1
        )
        return EncodingRegister(encoding)

    def get_objective_register(self) -> ObjectiveRegister:
        """
        Defines the default objective to minimize.
        """
        return ObjectiveRegister(
            objective_sense=ModeOptim.MINIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc={
                "nb_stations": ObjectiveDoc(
                    type=TypeObjective.OBJECTIVE, default_weight=1
                ),
                "penalty_overtime": ObjectiveDoc(
                    type=TypeObjective.PENALTY, default_weight=10000
                ),
                "penalty_undertime":
                # This is really soft, solver should handle it lexicographically.
                ObjectiveDoc(type=TypeObjective.PENALTY, default_weight=0),
                "penalty_precedence": ObjectiveDoc(
                    type=TypeObjective.PENALTY, default_weight=10000
                ),
            },
        )

    def get_solution_type_member(self) -> SalbpSolution:
        # don't satisfy the constraints ;)
        return SalbpSolution(self, allocation_to_station=[0] * self.nb_tasks)

    def get_graph_precedence(self) -> Graph:
        """
        Get precedence graph.

        Uses the PrecedenceProblem mixin's get_precedence_graph() method.
        """
        return self.get_precedence_graph()


def calculate_salbp_lower_bounds(problem: SalbpProblem) -> int:
    """
    Calculates lower bounds for the SALBP-1 problem.
    Inspired by: https://github.com/domain-independent-dp/didp-rs
    """
    times = [problem.task_times[t] for t in problem.tasks]
    c = problem.cycle_time

    # Bound 1: Theoretical minimum based on total work
    lb1 = math.ceil(sum(times) / c)

    # Bound 2: Based on tasks > c/2
    # Two tasks > c/2 cannot share a station.
    # One task == c/2 needs half a station.
    w2_1 = sum(1 for t in times if t > c / 2)
    w2_2 = sum(1 for t in times if t == c / 2)
    lb2 = w2_1 + math.ceil(w2_2 / 2)

    # Bound 3: More complex weighting
    # Tasks > 2c/3 count as 1
    # Tasks == 2c/3 count as 2/3
    # Tasks > c/3 count as 1/2
    # Tasks == c/3 count as 1/3
    w3 = 0
    for t in times:
        if t > c * 2 / 3:
            w3 += 1.0
        elif t == c * 2 / 3:
            w3 += 0.666
        elif t > c / 3:
            w3 += 0.5
        elif t == c / 3:
            w3 += 0.333
    lb3 = math.ceil(w3)

    return max(lb1, lb2, lb3)


class SalbpProblem_1_2(SalbpProblem):
    """
    Generalisation of SALBP-(1,2) problem.

    In this variant, both cycle time and number of stations can be optimized.
    """

    @staticmethod
    def from_salbp1(problem: SalbpProblem) -> "SalbpProblem_1_2":
        """Create SALBP-1,2 problem from SALBP-1 problem."""
        return SalbpProblem_1_2(
            tasks_data=problem.tasks_data,
            precedences=problem.precedence,
            number_of_stations=problem.nb_stations,
        )

    def __init__(
        self,
        tasks_data: List[TaskData],
        precedences: List[Tuple[int, int]],
        number_of_stations: int = None,
    ):
        """
        Initialize SALBP-1,2 problem (no fixed cycle time).

        Args:
            tasks_data: List of TaskData objects for each task
            precedences: List of (predecessor, successor) pairs
            number_of_stations: Number of available stations (default: nb_tasks)
        """
        # Call parent with cycle_time=None (will be optimized)
        super().__init__(
            tasks_data=tasks_data,
            cycle_time=None,
            precedences=precedences,
            number_of_stations=number_of_stations,
        )

    def evaluate(self, variable: SalbpSolution) -> Dict[str, float]:
        """
        Evaluate the solution.
        Objective: Minimize the number of stations and cycle time
        """
        stations_time_cumul = {}
        penalty_precedence = 0
        for i in range(self.nb_tasks):
            station_task_i = variable.allocation_to_station[i]
            if station_task_i not in stations_time_cumul:
                stations_time_cumul[station_task_i] = 0
            stations_time_cumul[station_task_i] += self.task_times[self.tasks_list[i]]
        for p, s in self.precedence:
            station_p = variable.get_start_time(task=p)
            station_s = variable.get_end_time(task=s)
            if station_p is not None and station_s is not None:
                if station_p > station_s:
                    penalty_precedence += 1

        nb_stations = len(stations_time_cumul)
        return {
            "nb_stations": nb_stations,
            "cycle_time": max(stations_time_cumul.values()),
            "cycle_time_dispersion": max(stations_time_cumul.values())
            - min(stations_time_cumul.values()),
            "penalty_precedence": penalty_precedence,
        }

    def satisfy(self, variable: SalbpSolution) -> bool:
        """
        Strict check for validity.
        """
        eval_res = self.evaluate(variable)
        return eval_res["penalty_precedence"] == 0

    def get_objective_register(self) -> ObjectiveRegister:
        """
        Defines the default objective to minimize.
        """
        return ObjectiveRegister(
            objective_sense=ModeOptim.MINIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc={
                "nb_stations": ObjectiveDoc(
                    type=TypeObjective.OBJECTIVE, default_weight=1
                ),
                "cycle_time": ObjectiveDoc(
                    type=TypeObjective.OBJECTIVE, default_weight=1
                ),
                "cycle_time_dispersion":
                # This is really soft, solver should handle it lexicographically.
                ObjectiveDoc(type=TypeObjective.PENALTY, default_weight=0),
                "penalty_precedence": ObjectiveDoc(
                    type=TypeObjective.PENALTY, default_weight=10000
                ),
            },
        )
