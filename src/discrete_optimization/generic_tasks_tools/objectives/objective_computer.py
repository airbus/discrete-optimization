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
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import (
    Objective,
)


class ObjectiveComputer(ABC, Generic[Task]):
    problem: TasksProblem[Task]

    def __init__(
        self, problem: TasksProblem[Task] = None, weight_objective: float = 1.0
    ):
        self.problem = problem
        self.weight_objective = weight_objective

    @staticmethod
    def get_objective_name() -> Objective | str: ...

    def set_problem(self, problem: TasksProblem[Task]):
        self.problem = problem

    @abstractmethod
    def compute_objective(self, solution: TasksSolution) -> float: ...
    @property
    def weight_cost(self) -> float:
        return self.weight_objective
