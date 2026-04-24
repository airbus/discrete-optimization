from __future__ import annotations

import logging

from discrete_optimization.generic_tasks_tools.base import (
    Task,
)
from discrete_optimization.generic_tasks_tools.precedence import (
    PrecedenceProblem,
    PrecedenceSolution,
)
from discrete_optimization.generic_tasks_tools.scheduling import (
    SchedulingProblem,
    SchedulingSolution,
)

logger = logging.Logger(__name__)


class PrecedenceSchedulingProblem(PrecedenceProblem[Task], SchedulingProblem[Task]):
    """Scheduling problem with precedence constraints on tasks."""

    ...


class PrecedenceSchedulingSolution(PrecedenceSolution[Task], SchedulingSolution[Task]):
    """Solution for scheduling problem with precedence constraints.

    Can implement `check_tasks_order` by using start and end times.

    """

    problem: PrecedenceSchedulingProblem[Task]

    def check_tasks_order(self, task1, task2) -> bool:
        """Check whether task1 is performed before task2.

        Args:
            task1:
            task2:

        Returns:
            True if task1 is finished before task2 starts, False else.

        """
        return self.get_end_time(task1) <= self.get_start_time(task2)
