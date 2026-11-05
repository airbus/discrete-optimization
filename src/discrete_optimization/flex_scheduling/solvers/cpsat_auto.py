#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import numpy as np

from discrete_optimization.flex_scheduling.problem import FlexProblem, ScheduleSolution
from discrete_optimization.generic_tasks_tools.allocation import UnaryResource
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import (
    RawSolution,
)
from discrete_optimization.generic_tasks_tools.skill import (
    Skill,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.auto import (
    GenericSchedulingAutoCpSatSolver,
)


class CpSatAutoFlexSchedulingSolver(GenericSchedulingAutoCpSatSolver):
    problem: FlexProblem

    def convert_task_variables_to_solution(
        self, raw_sol: RawSolution[Task, UnaryResource, Skill]
    ) -> ScheduleSolution:
        schedule = np.zeros((self.problem.nb_tasks, 2))
        allocation = np.zeros(self.problem.nb_tasks)
        for i in range(self.problem.nb_tasks):
            t = self.problem.tasks_list[i]
            schedule[i, 0] = raw_sol.task_variables[t].start
            schedule[i, 1] = raw_sol.task_variables[t].end
            allocation[i] = raw_sol.task_variables[t].mode
        return ScheduleSolution(
            problem=self.problem, schedule=schedule, modes=allocation
        )
