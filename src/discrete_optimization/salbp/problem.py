#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import Dict, List, Tuple

from discrete_optimization.generic_tasks_tools.allocation import (
    AllocationProblem,
    AllocationSolution,
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.scheduling import (
    SchedulingProblem,
    SchedulingSolution,
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

Task = int
Resource = int


class SalbpSolution(SchedulingSolution[Task], AllocationSolution[Task, Resource]):
    # We can see this as both a scheduling and allocation solution
    problem: "SalbpProblem"

    def get_end_time(self, task: Task) -> int:
        return self.allocation_to_station[self.problem.tasks_to_index[task]] + 1

    def get_start_time(self, task: Task) -> int:
        return self.allocation_to_station[self.problem.tasks_to_index[task]]

    def is_allocated(self, task: Task, unary_resource: UnaryResource) -> bool:
        return (
            self.allocation_to_station[self.problem.tasks_to_index[task]]
            == unary_resource
        )

    def __init__(self, problem: "SalbpProblem", allocation_to_station: list[int]):
        super().__init__(problem)
        self.task_alloc = []
        self.allocation_to_station = allocation_to_station
        self._nb_stations = len(set(self.allocation_to_station))

    def copy(self) -> "SalbpSolution":
        return SalbpSolution(self.problem, deepcopy(self.allocation_to_station))

    def change_problem(self, new_problem: "Problem") -> "Solution":
        return SalbpSolution(new_problem, deepcopy(self.allocation_to_station))

    def __str__(self):
        return f"SalbpSolution(nb_stations={self._nb_stations})"

    def __hash__(self):
        # Hash based on the sorted tuple of items to ensure determinism
        return hash(tuple(self.allocation_to_station))

    def __eq__(self, other):
        return self.allocation_to_station == other.allocation_to_station


class SalbpProblem(SchedulingProblem[Task], AllocationProblem[Task, Resource]):
    def get_makespan_upper_bound(self) -> int:
        # Max number of station, can be better.
        return self.number_of_tasks

    @property
    def unary_resources_list(self) -> list[UnaryResource]:
        return list(range(self.number_of_tasks))

    @property
    def tasks_list(self) -> list[Task]:
        return self.tasks

    def __init__(
        self,
        number_of_tasks: int,
        cycle_time: int,
        task_times: Dict[int, int],
        precedence: List[Tuple[int, int]],
    ):
        self.number_of_tasks = number_of_tasks
        self.cycle_time = cycle_time
        self.task_times = task_times
        self.precedence = precedence

        # Build adjacency for easier graph traversal
        self.adj = {i: [] for i in task_times.keys()}
        self.predecessors = {i: [] for i in task_times.keys()}
        for p, s in precedence:
            if p in self.adj:
                self.adj[p].append(s)
            if s in self.predecessors:
                self.predecessors[s].append(p)

        # Standard ordering of task IDs (assuming 1 to N usually, but we use keys)
        self.tasks = sorted(list(task_times.keys()))
        self.tasks_to_index = {self.tasks[i]: i for i in range(len(self.tasks))}

    def evaluate(self, variable: SalbpSolution) -> Dict[str, float]:
        """
        Evaluate the solution.
        Objective: Minimize the number of stations.
        Constraint Penalties: Over-cycle time, Precedence violations.
        """
        stations_time_cumul = {}
        penalty_overtime = 0
        penalty_undertime = 0
        penalty_precedence = 0
        # 1. Group by station and check Cycle Time
        for i in range(self.number_of_tasks):
            station_task_i = variable.allocation_to_station[i]
            if station_task_i not in stations_time_cumul:
                stations_time_cumul[station_task_i] = 0
            stations_time_cumul[station_task_i] += self.task_times[self.tasks_list[i]]

        for s_id, total_time in stations_time_cumul.items():
            penalty_overtime += max(0, total_time - self.cycle_time)
            penalty_undertime += max(0, self.cycle_time - total_time)

        # 2. Check Precedence: Pred station <= Succ station
        for p, s in self.precedence:
            station_p = variable.get_start_time(task=p)
            station_s = variable.get_end_time(task=s)

            # If tasks are missing from solution, that's a structural issue,
            # usually assumed complete in a valid solution object.
            if station_p is not None and station_s is not None:
                if station_p > station_s:
                    penalty_precedence += 1

        nb_stations = len(stations_time_cumul)
        # Total cost logic (Soft constraints handling)
        # We assume standard objective is just nb_stations,
        # but for solvers we often add penalties.
        return {
            "nb_stations": nb_stations,
            "penalty_overtime": penalty_overtime,
            "penalty_undertime": penalty_undertime,
            "penalty_precedence": penalty_precedence,
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
            length=self.number_of_tasks, lows=0, ups=self.number_of_tasks - 1
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
        return SalbpSolution(self, allocation_to_station=[0] * self.number_of_tasks)
