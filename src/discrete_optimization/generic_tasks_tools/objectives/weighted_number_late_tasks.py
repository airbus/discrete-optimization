#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tasks_tools.batching import BatchingProblem
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import Objective
from discrete_optimization.generic_tasks_tools.objectives.objective_computer import (
    ObjectiveComputer,
)
from discrete_optimization.generic_tasks_tools.scheduling import (
    SchedulingProblem,
    SchedulingSolution,
    Task,
)


class WeightedNumberLateComputer(ObjectiveComputer[Task]):
    problem: SchedulingProblem[Task]

    @staticmethod
    def get_objective_name() -> Objective | str:
        return Objective.WEIGHTED_NB_LATE_TASKS

    def __init__(
        self,
        problem: SchedulingProblem[Task],
        weight_objective: float = 1.0,
        max_end_and_weight: dict[Task, tuple[int | None, int | None]] = None,
    ):
        super().__init__(problem, weight_objective)
        if max_end_and_weight is None:
            self.max_end_and_weight = {}
        else:
            self.max_end_and_weight = max_end_and_weight

    def compute_objective(self, solution: SchedulingSolution) -> float:
        val = 0
        for task in self.max_end_and_weight:
            if solution.is_present(task):
                end = solution.get_end_time(task)
                if end > self.max_end_and_weight[task][0]:
                    val += self.max_end_and_weight[task][1]
        return val


class WeightedNumberLateTasksInBatchComputer(WeightedNumberLateComputer[Task]):
    problem: BatchingProblem[Task]

    @staticmethod
    def get_objective_name() -> Objective | str:
        return Objective.WEIGHTED_NB_LATE_TASKS_BATCH
