#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from collections.abc import Callable

from discrete_optimization.generic_tasks_tools.base import TasksProblem, TasksSolution
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import Objective
from discrete_optimization.generic_tasks_tools.multimode import (
    Task,
)
from discrete_optimization.generic_tasks_tools.objectives.objective_computer import (
    ObjectiveComputer,
)


class CustomComputer(ObjectiveComputer[Task]):
    @staticmethod
    def get_objective_name() -> str:
        return Objective.CUSTOM

    def __init__(
        self,
        problem: TasksProblem[Task] | None = None,
        weight_objective: float | int = 1,
        custom_fn: Callable[[TasksProblem[Task], TasksSolution[Task]], float | int]
        | None = None,
    ):
        super().__init__(problem, weight_objective)
        if custom_fn is None:
            raise ValueError("custom_fn cannot be None")
        else:
            self.custom_fn = custom_fn

    def compute_objective(self, solution: TasksSolution[Task]) -> float | int:
        return self.custom_fn(self.problem, solution)
