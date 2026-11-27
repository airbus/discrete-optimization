#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from abc import ABC, abstractmethod
from typing import Generic

from discrete_optimization.generic_tasks_tools.base import (
    Task,
    TasksProblem,
    TasksSolution,
)


class ObjectiveComputer(ABC, Generic[Task]):
    problem: TasksProblem[Task]

    def __init__(
        self,
        problem: TasksProblem[Task] | None = None,
        weight_objective: float | int = 1,
    ):
        self.weight_objective = weight_objective
        if problem is not None:
            self.set_problem(problem)

    @staticmethod
    def get_objective_name() -> str: ...

    def set_problem(self, problem: TasksProblem[Task]):
        self.problem = problem

    @abstractmethod
    def compute_objective(self, solution: TasksSolution[Task]) -> float | int: ...
    @property
    def weight_cost(self) -> float | int:
        return self.weight_objective
