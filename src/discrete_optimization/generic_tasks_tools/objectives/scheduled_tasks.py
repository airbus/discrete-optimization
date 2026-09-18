#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Generic

from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import Objective
from discrete_optimization.generic_tasks_tools.objectives.objective_computer import (
    ObjectiveComputer,
)
from discrete_optimization.generic_tasks_tools.scheduling import (
    SchedulingProblem,
    SchedulingSolution,
    Task,
)


class ScheduledTasksComputer(ObjectiveComputer[Task], Generic[Task]):
    problem: SchedulingProblem[Task]

    def __init__(
        self,
        problem: SchedulingProblem[Task],
        weight_objective: float = 1.0,
        weight_per_task: dict[Task, int] | None = None,
    ):
        super().__init__(problem=problem, weight_objective=weight_objective)
        if weight_per_task is None:
            self.weight_per_task = {t: 1 for t in self.problem.tasks_list}
        else:
            self.weight_per_task = weight_per_task

    def get_objective_name(self) -> Objective | str:
        return Objective.NB_TASKS_ALLOCATED

    def compute_objective(self, solution: SchedulingSolution[Task]) -> float:
        return sum(
            self.weight_per_task[t] * solution.is_present(t)
            for t in self.weight_per_task
        )
