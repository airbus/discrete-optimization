#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import Objective
from discrete_optimization.generic_tasks_tools.multimode import (
    MultimodeProblem,
    MultimodeSolution,
    Task,
)
from discrete_optimization.generic_tasks_tools.objectives.objective_computer import (
    ObjectiveComputer,
)


class ModeCostComputer(ObjectiveComputer[Task]):
    problem: MultimodeProblem[Task]

    @staticmethod
    def get_objective_name() -> Objective | str:
        return Objective.MODE_COST

    def __init__(
        self,
        problem: MultimodeProblem[Task] = None,
        weight_objective: float = 1.0,
        mode_cost: dict[tuple[Task, int], int] = None,
    ):
        super().__init__(problem, weight_objective)
        if mode_cost is None:
            self._mode_cost = {}
        else:
            self._mode_cost = mode_cost

    def mode_cost(self, task: Task, mode: int) -> float:
        return self._mode_cost.get((task, mode), 0)

    def has_any_mode_cost(self):
        return any(self._mode_cost[tm] != 0 for tm in self._mode_cost)

    def compute_objective(self, solution: MultimodeSolution[Task]) -> float:
        return sum(
            self.mode_cost(task=task, mode=solution.get_mode(task))
            for task in self.problem.tasks_list
        )
