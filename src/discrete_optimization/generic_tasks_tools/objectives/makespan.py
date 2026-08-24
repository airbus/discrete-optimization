#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import Objective
from discrete_optimization.generic_tasks_tools.objectives.objective_computer import (
    ObjectiveComputer,
)
from discrete_optimization.generic_tasks_tools.scheduling import (
    SchedulingProblem,
    SchedulingSolution,
    Task,
)


class MakespanObjectiveComputer(ObjectiveComputer[Task]):
    problem: SchedulingProblem[Task]

    @staticmethod
    def get_objective_name() -> Objective | str:
        return Objective.MAKESPAN

    def compute_objective(self, solution: SchedulingSolution[Task]) -> int:
        return solution.get_max_end_time()
