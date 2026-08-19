#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from dataclasses import dataclass
from typing import Generic, Hashable, Optional

import networkx as nx

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.multimode_scheduling import (
    MultimodeSchedulingProblem,
    MultimodeSchedulingSolution,
)

logger = logging.getLogger(__name__)


@dataclass
class AlternativeSchedulingSubProblem:
    source_task: Hashable  # (mandatory task from which originates the alternative)
    sink_task: Hashable  # (mandatory task from which finished the alternative)
    graph: Optional[nx.DiGraph] = None
    is_graph_successors: bool = True
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


class AlternativeSchedulingProblem(MultimodeSchedulingProblem[Task], Generic[Task]):
    # @abstractmethod
    def get_alternative_scheduling_subproblem(
        self,
    ) -> list[AlternativeSchedulingSubProblem]:
        return []


class NoAlternativeSchedulingProblem(AlternativeSchedulingProblem[Task]):
    def get_alternative_scheduling_subproblem(
        self,
    ) -> list[AlternativeSchedulingSubProblem]:
        return []


class AlternativeSchedulingSolution(MultimodeSchedulingSolution[Task]):
    problem: AlternativeSchedulingProblem[Task]

    def check_alternative_scheduling_subproblem(self) -> None:
        for alt_problem in self.problem.get_alternative_scheduling_subproblem():
            paths = alt_problem.list_paths
            nb_path_done = 0
            paths_done = []
            for p in paths:
                if all(self.is_present(task) for task in p):
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
                        if not (self.get_end_time(t0) <= self.get_start_time(t1)):
                            logger.info(
                                "Precedence constraints not respected in the final schedule"
                            )
                            logger.info(f"between {t1} and {t0}")
                            return False
        return True
