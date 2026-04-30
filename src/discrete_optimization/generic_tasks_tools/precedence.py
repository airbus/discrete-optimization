from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Iterable

from discrete_optimization.generic_tasks_tools.base import (
    Task,
    TasksProblem,
    TasksSolution,
)
from discrete_optimization.generic_tools.graph_api import Graph

logger = logging.getLogger(__name__)


class PrecedenceProblem(TasksProblem[Task]):
    """Problem with precedence constraints on tasks."""

    @abstractmethod
    def get_precedence_constraints(self) -> dict[Task, Iterable[Task]]:
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


class PrecedenceSolution(TasksSolution[Task]):
    """Solution for problem with precedence constraints."""

    problem: PrecedenceProblem[Task]

    @abstractmethod
    def check_tasks_order(self, task1, task2) -> bool:
        """Check whether task1 is performed before task2.

        Args:
            task1:
            task2:

        Returns:
            True if task1 is finished before task2 starts, False else.

        """
        ...

    def check_precedence_constraints(self) -> bool:
        """Check that all precedence constraints are satisfied.

        Returns:

        """
        for task1, successors in self.problem.get_precedence_constraints().items():
            for task2 in successors:
                if not self.check_tasks_order(task1, task2):
                    logger.debug(
                        f"Precedence relationship broken: {task1} ends after {task2} starts."
                    )
                    return False

        return True


class WithoutPrecedenceProblem(PrecedenceProblem[Task]):
    """Utility mixin for problem w/o precedence constraints.

    To be used has an additional mixin with generic `AllocationSchedulingProblem`.

    """

    def get_precedence_constraints(self) -> dict[Task, Iterable[Task]]:
        return {}
