#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Generic

from discrete_optimization.generic_tasks_tools.allocation import (
    AllocationProblem,
    AllocationSolution,
    Task,
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import Objective
from discrete_optimization.generic_tasks_tools.objectives.objective_computer import (
    ObjectiveComputer,
)


class AllocatedTasksObjective(ObjectiveComputer[Task], Generic[Task, UnaryResource]):
    problem: AllocationProblem[Task, UnaryResource]

    @staticmethod
    def get_objective_name() -> Objective | str:
        return Objective.NB_TASKS_DONE

    def compute_objective(self, solution: AllocationSolution[Task, UnaryResource]):
        return solution.compute_nb_tasks_done()
