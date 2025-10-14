from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from discrete_optimization.generic_tools.cp_tools import CpSolver, SignEnum
from discrete_optimization.generic_tools.do_problem import Problem, Solution

Task = TypeVar("Task")


class TasksProblem(Problem, Generic[Task]):
    """Base class for scheduling/allocation problems."""

    tasks_list: list[Task]
    """List of all tasks to schedule or allocate to."""


class TasksSolution(ABC, Solution, Generic[Task]):
    """Base class for sheduling/allocation solutions."""

    problem: TasksProblem[Task]


class TasksCpSolver(CpSolver, Generic[Task]):
    """Base class for cp solver handling tasks problems."""

    problem: TasksProblem[Task]
