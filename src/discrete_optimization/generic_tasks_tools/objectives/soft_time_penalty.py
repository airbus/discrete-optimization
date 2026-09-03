#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Generic

from discrete_optimization.generic_tasks_tools.enums import MinOrMax, StartOrEnd
from discrete_optimization.generic_tasks_tools.generic_scheduling import (
    GenericSchedulingProblem,
    GenericSchedulingSolution,
    NonRenewableResource,
    NonSkillCumulativeResource,
    Skill,
    Task,
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import Objective
from discrete_optimization.generic_tasks_tools.objectives.objective_computer import (
    ObjectiveComputer,
)


class SoftTimePenaltyComputer(
    ObjectiveComputer[Task],
    Generic[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, NonRenewableResource
    ],
):
    problem: GenericSchedulingProblem[
        Task, UnaryResource, Skill, NonSkillCumulativeResource, NonRenewableResource
    ]
    penalty = 0

    @staticmethod
    def get_objective_name() -> Objective:
        return Objective.TIME_PENALTY

    def compute_objective(
        self,
        solution: GenericSchedulingSolution[
            Task, UnaryResource, Skill, NonSkillCumulativeResource, NonRenewableResource
        ],
    ) -> float:
        penalty = 0
        # time windows
        for task in self.problem.tasks_list:
            start = solution.get_start_time(task)
            end = solution.get_end_time(task)
            start_lb = self.problem.get_task_start_or_end_lower_bound(
                task=task, start_or_end=StartOrEnd.START
            )
            end_lb = self.problem.get_task_start_or_end_lower_bound(
                task=task, start_or_end=StartOrEnd.END
            )
            start_ub = self.problem.get_task_start_or_end_upper_bound(
                task=task, start_or_end=StartOrEnd.START
            )
            end_ub = self.problem.get_task_start_or_end_upper_bound(
                task=task, start_or_end=StartOrEnd.END
            )
            penalty += max(0, start_lb - start)
            penalty += max(0, end_lb - end)
            penalty += max(0, start - start_ub)
            penalty += max(0, end - end_ub)
        # time lags
        for task1_start_or_end in StartOrEnd:
            for task2_start_or_end in StartOrEnd:
                for min_or_max in MinOrMax:
                    for task1, task2, offset in self.problem.get_original_time_lags(
                        task1_start_or_end=task1_start_or_end,
                        task2_start_or_end=task2_start_or_end,
                        min_or_max=min_or_max,
                    ):
                        t1 = solution.get_start_or_end_time(
                            task=task1, start_or_end=task1_start_or_end
                        )
                        t2 = solution.get_start_or_end_time(
                            task=task2, start_or_end=task2_start_or_end
                        )
                        if min_or_max == MinOrMax.MIN:
                            penalty += max(0, t1 + offset - t2)
                        else:
                            penalty += max(0, t2 - (t1 + offset))
        return penalty
