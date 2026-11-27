#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from dataclasses import dataclass, field
from typing import Generic, Hashable, Optional

import networkx as nx

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.multimode_scheduling import (
    MultimodeSchedulingProblem,
    MultimodeSchedulingSolution,
)
from discrete_optimization.generic_tasks_tools.utils import optional_override

logger = logging.getLogger(__name__)


@dataclass
class AlternativeSchedulingSubProblem:
    source_task: Hashable = (
        None  # (optional task from which originates the alternative)
    )
    sink_task: Hashable = None  # (optional task from which finished the alternative)
    graph: Optional[nx.DiGraph] = None
    is_graph_successors: bool = True
    list_paths: Optional[list[list[Hashable]]] = None
    is_path_successors: bool = True
    nb_path_to_do: int = 1  # by default 1 subpath to follow.
    has_a_source: bool = field(init=False)
    has_a_sink: bool = field(init=False)

    def __post_init__(self):
        self.has_a_source = self.source_task is not None
        self.has_a_sink = self.sink_task is not None
        if self.graph is None:
            if self.list_paths:
                graph = nx.DiGraph()
                if self.has_a_source:
                    graph.add_node(self.source_task)
                if self.has_a_sink:
                    graph.add_node(self.sink_task)
                for p in self.list_paths:
                    for e0, e1 in zip(p[:-1], p[1:]):
                        graph.add_edge(e0, e1)
                    if self.has_a_source:
                        if p[0] != self.source_task:
                            graph.add_edge(self.source_task, p[0])
                    if self.has_a_sink:
                        if p[-1] != self.sink_task:
                            graph.add_edge(self.source_task, p[0])
                self.graph = graph
        if self.has_a_source and self.source_task not in self.graph.nodes:
            predecessors = {
                n: list(self.graph.predecessors(n)) for n in self.graph.nodes
            }
            self.graph.add_node(self.source_task)
            for n in predecessors:
                if len(predecessors[n]) == 0:
                    self.graph.add_edge(self.source_task, n)
        if self.has_a_sink and self.sink_task not in self.graph.nodes:
            successors = {n: list(self.graph.successors(n)) for n in self.graph.nodes}
            self.graph.add_node(self.sink_task)
            for n in successors:
                if len(successors[n]) == 0:
                    self.graph.add_edge(n, self.sink_task)
        if self.list_paths is None:
            if self.has_a_source and self.has_a_sink:
                self.list_paths = list(
                    nx.all_simple_paths(self.graph, self.source_task, self.sink_task)
                )
            else:
                # let's find node without incoming edge.
                sources = {
                    n
                    for n in self.graph.nodes
                    if len(list(self.graph.predecessors(n))) == 0
                }
                sinks = {
                    n
                    for n in self.graph.nodes
                    if len(list(self.graph.successors(n))) == 0
                }
                self.list_paths = []
                for source in sources:
                    for sink in sinks:
                        self.list_paths.extend(
                            list(nx.all_simple_paths(self.graph, source, sink))
                        )
                logger.info(f"{self.list_paths}")


class AlternativeSchedulingProblem(MultimodeSchedulingProblem[Task], Generic[Task]):
    @optional_override
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
