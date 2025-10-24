#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import math
import random
from abc import abstractmethod
from typing import Generic, Optional, Union

import numpy as np

from discrete_optimization.generic_tasks_tools.base import (
    Task,
    TasksProblem,
    TasksSolution,
)
from discrete_optimization.generic_tasks_tools.precedence import (
    HashableTask,
    PrecedenceProblem,
)
from discrete_optimization.generic_tasks_tools.scheduling import (
    SchedulingProblem,
    SchedulingSolution,
)
from discrete_optimization.generic_tools.graph_api import Graph

logger = logging.getLogger(__name__)


class NeighborBuilder(Generic[Task]):
    problem: TasksProblem[Task]

    @abstractmethod
    def find_subtasks(
        self,
        current_solution: TasksSolution[Task],
        subtasks: Optional[set[Task]] = None,
    ) -> tuple[set[Task], set[Task]]:
        """
        Split the scheduling task set in 2 part, it can then be used by constraint handler to introduce different
        constraints in those two subsets. Usually the first returned set will be considered
        like the subproblem in LNS
        Args:
            current_solution: current solution to consider
            subtasks: possibly existing subset of tasks that are in the neighborhood

        Returns:

        """
        ...


def intersect(i1: tuple[int, int], i2: tuple[int, int]):
    if i2[0] >= i1[1] or i1[0] >= i2[1]:
        return None
    else:
        s = max(i1[0], i2[0])
        e = min(i1[1], i2[1])
        return [s, e]


class NeighborBuilderSubPart(NeighborBuilder[Task]):
    """
    Cut the schedule in different subpart in the increasing order of the schedule.
    """

    def __init__(self, problem: TasksProblem[Task], nb_cut_part: int = 10):
        self.problem = problem
        self.nb_cut_part = nb_cut_part
        self.current_sub_part = 0
        self.set_tasks = set(self.problem.tasks_list)

    def find_subtasks(
        self,
        current_solution: TasksSolution[Task],
        subtasks: Optional[set[Task]] = None,
    ) -> tuple[set[Task], set[Task]]:
        n_jobs = len(self.problem.tasks_list)
        nb_job_sub = math.ceil(n_jobs / self.nb_cut_part)
        if isinstance(current_solution, SchedulingSolution):
            task_of_interest = sorted(
                self.problem.tasks_list, key=lambda x: current_solution.get_end_time(x)
            )
        else:
            task_of_interest = self.problem.tasks_list
        if self.current_sub_part * nb_job_sub >= n_jobs:
            # For small problems, we can have some border effect, finally leading to a bug
            # in the subsequent code (if len(subtasks) == 0:
            #                           subtasks = [task_of_interest[-1]])
            task_of_interest = task_of_interest[-nb_job_sub:]
        else:
            task_of_interest = task_of_interest[
                self.current_sub_part * nb_job_sub : (self.current_sub_part + 1)
                * nb_job_sub
            ]
        if subtasks is None:
            subtasks = task_of_interest
        else:
            subtasks.update(task_of_interest)
        if len(subtasks) == 0:
            subtasks = [task_of_interest[-1]]
        self.current_sub_part = (self.current_sub_part + 1) % self.nb_cut_part
        return subtasks, self.set_tasks.difference(subtasks)


class NeighborRandom(NeighborBuilder[Task]):
    def __init__(
        self,
        problem: TasksProblem[Task],
        fraction_subproblem: float = 0.9,
        delta_abs_time_from_makespan_to_not_fix: int = 5,
        delta_rel_time_from_makespan_to_not_fix: float = 0.0,
    ):
        self.problem = problem
        self.fraction_subproblem = fraction_subproblem
        self.delta_abs_time_from_makespan_to_not_fix = (
            delta_abs_time_from_makespan_to_not_fix
        )
        self.delta_rel_time_from_makespan_to_not_fix = (
            delta_rel_time_from_makespan_to_not_fix
        )
        self.set_tasks = set(self.problem.tasks_list)

    def find_subtasks(
        self,
        current_solution: TasksSolution[Task],
        subtasks: Optional[set[Task]] = None,
    ) -> tuple[set[Task], set[Task]]:
        if subtasks is None:
            subtasks = set()

        nb_jobs = len(self.problem.tasks_list)
        tasks_subproblem = set(
            random.sample(
                self.problem.tasks_list, int(self.fraction_subproblem * nb_jobs)
            )
        )

        if isinstance(current_solution, SchedulingSolution):
            max_time = current_solution.get_max_end_time()
            last_jobs = [
                x
                for x in self.problem.tasks_list
                if max_time - self.delta_abs_time_from_makespan_to_not_fix
                <= current_solution.get_end_time(x)
                or (1 - self.delta_rel_time_from_makespan_to_not_fix) * max_time
                <= current_solution.get_end_time(x)
            ]
            for lj in last_jobs:
                if lj not in tasks_subproblem:
                    tasks_subproblem.add(lj)

        subtasks.update(tasks_subproblem)
        return subtasks, self.set_tasks.difference(subtasks)


class NeighborBuilderMix(NeighborBuilder[Task]):
    def __init__(
        self,
        list_neighbor: list[NeighborBuilder[Task]],
        weight_neighbor: Union[list[float], np.array],
        verbose: bool = False,
    ):
        self.list_neighbor = list_neighbor
        self.weight_neighbor = weight_neighbor
        if isinstance(self.weight_neighbor, list):
            self.weight_neighbor = np.array(self.weight_neighbor)
        self.weight_neighbor = self.weight_neighbor / np.sum(self.weight_neighbor)
        self.index_np = np.array(range(len(self.list_neighbor)), dtype=np.int_)
        self.verbose = verbose

    def find_subtasks(
        self,
        current_solution: TasksSolution[Task],
        subtasks: Optional[set[Task]] = None,
    ) -> tuple[set[Task], set[Task]]:
        choice = np.random.choice(self.index_np, size=1, p=self.weight_neighbor)[0]
        return self.list_neighbor[choice].find_subtasks(
            current_solution=current_solution, subtasks=subtasks
        )


class NeighborBuilderTimeWindow(NeighborBuilder[Task]):
    def __init__(self, problem: TasksProblem[Task], time_window_length: int = 10):
        self.problem = problem
        self.time_window_length = time_window_length
        self.current_time_window = [0, self.time_window_length]

    def find_subtasks(
        self,
        current_solution: TasksSolution[Task],
        subtasks: Optional[set[Task]] = None,
    ) -> tuple[set[Task], set[Task]]:
        if not isinstance(current_solution, SchedulingSolution):
            raise ValueError(
                "This neighbor builder is applicable only to a scheduling solution."
            )
        last_time = current_solution.get_max_end_time()
        if self.current_time_window[0] >= last_time:
            self.current_time_window = [0, self.time_window_length]
        tasks_of_interest = [
            t
            for t in self.problem.tasks_list
            if any(
                current_solution.get_start_time(t)
                <= x
                <= current_solution.get_end_time(t)
                for x in range(self.current_time_window[0], self.current_time_window[1])
            )
        ]
        other_tasks = [t for t in self.problem.tasks_list if t not in tasks_of_interest]
        self.current_time_window = [
            self.current_time_window[0] + self.time_window_length,
            self.current_time_window[1] + self.time_window_length,
        ]
        return set(tasks_of_interest), set(other_tasks)


class NeighborRandomAndNeighborGraph(NeighborBuilder[HashableTask]):
    def __init__(
        self,
        problem: TasksProblem[HashableTask],
        graph: Optional[Graph] = None,
        fraction_subproblem: float = 0.05,
    ):
        if not isinstance(problem, PrecedenceProblem):
            raise ValueError(
                "This neighbor builder is applicable only to a problem with precedence constraints."
            )
        self.problem = problem
        if graph is None:
            self.graph = problem.get_precedence_graph()
        else:
            self.graph = graph
        self.fraction_subproblem = fraction_subproblem
        self.nb_jobs_subproblem = math.ceil(
            len(self.problem.tasks_list) * self.fraction_subproblem
        )
        self.set_tasks = set(self.problem.tasks_list)

    def find_subtasks(
        self,
        current_solution: TasksSolution[HashableTask],
        subtasks: Optional[set[HashableTask]] = None,
    ) -> tuple[set[HashableTask], set[HashableTask]]:
        if not isinstance(current_solution, SchedulingSolution):
            raise ValueError(
                "This neighbor builder is applicable only to a scheduling solution."
            )
        if subtasks is None:
            subtasks = set()
            len_subtask = 0
        else:
            len_subtask = len(subtasks)
        while len_subtask < self.nb_jobs_subproblem:
            random_pick = random.choice(self.problem.tasks_list)
            interval = (
                current_solution.get_start_time(random_pick),
                current_solution.get_end_time(random_pick),
            )
            task_intersect = [
                t
                for t in self.problem.tasks_list
                if intersect(
                    interval,
                    (
                        current_solution.get_start_time(t),
                        current_solution.get_end_time(t),
                    ),
                )
                is not None
            ]
            for k in set(task_intersect):
                task_intersect += list(self.graph.get_predecessors(k)) + list(
                    self.graph.get_neighbors(k)
                )
            subtasks.update(task_intersect)
            len_subtask = len(subtasks)
        if len(subtasks) >= self.nb_jobs_subproblem:
            subtasks = set(random.sample(list(subtasks), self.nb_jobs_subproblem))
        return subtasks, self.set_tasks.difference(subtasks)


def build_default_neighbor_builder(
    problem: TasksProblem[Task],
) -> NeighborBuilder[Task]:
    if isinstance(problem, PrecedenceProblem) and isinstance(
        problem, SchedulingProblem
    ):
        return NeighborBuilderMix(
            list_neighbor=[
                NeighborBuilderSubPart(
                    problem=problem,
                ),
                NeighborRandomAndNeighborGraph(problem=problem),
            ],
            weight_neighbor=[0.5, 0.5],
        )
    else:
        return NeighborBuilderMix(
            list_neighbor=[
                NeighborBuilderSubPart(
                    problem=problem,
                ),
                NeighborRandom(problem=problem),
            ],
            weight_neighbor=[0.5, 0.5],
        )
