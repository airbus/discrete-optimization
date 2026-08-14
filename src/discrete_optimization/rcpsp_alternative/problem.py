#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  Implementation of RCPSP with optional alternative subproblems
#  Between 2 mandatory task, different subpath of task should be accomplished (with or without precedence constraints).
#  This problem can represent different alternative physical path of task to accomplish on a shop floor.
import logging
from dataclasses import dataclass
from typing import Any, Hashable, Optional, Union

import networkx as nx

from discrete_optimization.rcpsp.problem import RcpspProblem, RcpspSolution
from discrete_optimization.rcpsp.solution import RcpspSolution, Resource, Task
from discrete_optimization.rcpsp.special_constraints import (
    SpecialConstraintsDescription,
)

logger = logging.getLogger(__name__)


@dataclass
class AlternativeSchedulingSubProblem:
    source_task: Hashable  # (mandatory task from which originates the alternative)
    sink_task: Hashable  # (mandatory task from which finished the alternative)
    graph: Optional[nx.DiGraph] = None
    is_graph_successors: bool = True
    successors: dict[Hashable, list[Hashable]] = None
    list_paths: Optional[list[list[Hashable]]] = None
    is_path_successors: bool = True
    nb_path_to_do: int = 1  # by default 1 subpath to follow.

    def __post_init__(self):
        if self.graph is None:
            if self.list_paths:
                graph = nx.DiGraph()
                graph.add_node(self.source_task)
                graph.add_node(self.sink_task)
                for p in self.list_paths:
                    for e0, e1 in zip(p[:-1], p[1:]):
                        graph.add_edge(e0, e1)
                    if p[0] != self.source_task:
                        graph.add_edge(self.source_task, p[0])
                    if p[-1] != self.sink_task:
                        graph.add_edge(self.source_task, p[0])
                self.graph = graph
        if self.source_task not in self.graph.nodes:
            predecessors = {
                n: list(self.graph.predecessors(n)) for n in self.graph.nodes
            }
            self.graph.add_node(self.source_task)
            for n in predecessors:
                if len(predecessors[n]) == 0:
                    self.graph.add_edge(self.source_task, n)
        if self.sink_task not in self.graph.nodes:
            successors = {n: list(self.graph.successors(n)) for n in self.graph.nodes}
            self.graph.add_node(self.sink_task)
            for n in successors:
                if len(successors[n]) == 0:
                    self.graph.add_edge(n, self.sink_task)
        if self.list_paths is None:
            self.list_paths = list(
                nx.all_simple_paths(self.graph, self.source_task, self.sink_task)
            )


class RcpspWithAlternativePath(RcpspProblem):
    def __init__(
        self,
        resources: dict[str, Union[int, list[int]]],
        non_renewable_resources: list[str],
        mode_details: dict[Hashable, dict[int, dict[str, int]]],
        successors: dict[Hashable, list[Hashable]],
        horizon: int,
        tasks_list: Optional[list[Hashable]] = None,
        source_task: Optional[Hashable] = None,
        sink_task: Optional[Hashable] = None,
        name_task: Optional[dict[Hashable, str]] = None,
        calendar_details: Optional[dict[str, list[list[int]]]] = None,
        special_constraints: Optional[SpecialConstraintsDescription] = None,
        fixed_permutation: Optional[list[int]] = None,
        fixed_modes: Optional[list[int]] = None,
        alternative_tasks: Optional[list[Hashable]] = None,
        alternative_tasks_data: dict[Hashable, dict[int, dict[str, int]]] = None,
        alternative_successors: dict[Hashable, list[Hashable]] = None,
        list_alternative_subproblem: list[AlternativeSchedulingSubProblem] = None,
        **kwargs: Any,
    ):
        """
        Extension of RCPSPProblem, including
        :param alternative_tasks: tasks that are not mandatory
        :param alternative_tasks_data: data of the tasks when they are done (duration, resource usage),
         given per mode (like mode_details attribute)
        :param alternative_successors: successors of optional task (when active).
        The successors can be either optional or mandatory task.
        :param list_alternative_subproblem: list of alternative scheduling subproblem, describing the alternative paths.
        """
        self.alternative_tasks = alternative_tasks
        self.alternative_tasks_data = alternative_tasks_data
        self.alternative_successors = alternative_successors
        self.list_alternative_subproblem = list_alternative_subproblem
        super().__init__(
            resources=resources,
            non_renewable_resources=non_renewable_resources,
            mode_details=mode_details,
            successors=successors,
            horizon=horizon,
            tasks_list=tasks_list,
            source_task=source_task,
            sink_task=sink_task,
            name_task=name_task,
            calendar_details=calendar_details,
            special_constraints=special_constraints,
            fixed_permutation=fixed_permutation,
            fixed_modes=fixed_modes,
            **kwargs,
        )
        self.n_jobs = len(self.tasks_list)
        self.n_jobs_non_dummy = self.n_jobs - 2
        self.index_task = {self.tasks_list[i]: i for i in range(self.n_jobs)}
        self.tasks_list_non_dummy = [
            t for t in self.tasks_list if t not in {self.source_task, self.sink_task}
        ]
        self.index_task_non_dummy = {
            self.tasks_list_non_dummy[i]: i for i in range(self.n_jobs_non_dummy)
        }

    @property
    def tasks_list(self) -> list[Task]:
        return self._tasks_list + self.alternative_tasks

    def get_cumulative_resource_consumption(
        self, resource: Resource, task: Task, mode: int
    ) -> int:
        if task in self.mode_details:
            return self.mode_details[task][mode].get(resource, 0)
        if task in self.alternative_tasks_data:
            if mode is None:
                return 0
            return self.alternative_tasks_data[task][mode].get(resource, 0)
        return 0

    def get_task_mode_duration(self, task: Task, mode: int) -> int:
        if task in self._tasks_list:
            return self.mode_details[task][mode]["duration"]
        if task in self.alternative_tasks:
            return self.alternative_tasks_data[task][mode]["duration"]
        return 0

    def get_task_modes(self, task: Task) -> set[int]:
        if task in self._tasks_list:
            return set(self.mode_details[task])
        if task in self.alternative_tasks:
            return set(self.alternative_tasks_data[task])
        return None

    def satisfy(self, variable: RcpspSolution) -> bool:
        sat = super().satisfy(variable)
        if not sat:
            return False
        for alt_problem in self.list_alternative_subproblem:
            paths = alt_problem.list_paths
            nb_path_done = 0
            paths_done = []
            for p in paths:
                if all(
                    task in variable.rcpsp_schedule
                    and variable.get_mode(task) is not None
                    for task in p
                ):
                    nb_path_done += 1
                    paths_done.append(p)
            if nb_path_done > alt_problem.nb_path_to_do:
                logger.info("Too much alternative path")
                return False
            if nb_path_done < alt_problem.nb_path_to_do:
                logger.info("Not enough alternative path")
                return False
            if alt_problem.is_path_successors:
                for p in paths_done:
                    for t0, t1 in zip(p[:-1], p[1:]):
                        if not (
                            variable.get_end_time(t0) <= variable.get_start_time(t1)
                        ):
                            logger.info(
                                "Precedence constraints not respected in the final schedule"
                            )
                            logger.info(f"between {t1} and {t0}")
                            return False
        return True


def get_optional_tasks_done(sol: RcpspSolution, problem: RcpspWithAlternativePath):
    return [t for t in problem.alternative_tasks if sol.get_mode(t) is not None]
