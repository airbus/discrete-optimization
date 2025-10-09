from __future__ import annotations

from abc import abstractmethod
from collections.abc import Hashable, Iterable
from typing import TypeVar

from discrete_optimization.generic_tasks_tools.base import Task, TasksProblem
from discrete_optimization.generic_tools.graph_api import Graph

HashableTask = TypeVar("HashableTask", bound=Hashable)


class PrecedenceProblem(TasksProblem[HashableTask]):
    """Problem with precedence constraints on tasks."""

    @abstractmethod
    def get_precedence_constraints(self) -> dict[HashableTask, Iterable[HashableTask]]:
        """Map each task to the tasks that need to be performed after it."""
        ...

    def get_precedence_graph(self) -> Graph:
        nodes = [(task, {}) for task in self.tasks_list]
        edges = []
        successors = self.get_precedence_constraints()
        for n in successors:
            for succ in successors[n]:
                edges += [(n, succ, {})]
        return Graph(nodes, edges, False)
