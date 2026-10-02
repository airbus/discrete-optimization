#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Generic

from discrete_optimization.generic_tasks_tools.calendar_preemptive import (
    CalendarPreemptiveProblem,
    CalendarPreemptiveSolution,
    CumulativeResource,
    OtherCalendarResource,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling import Task
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import Objective
from discrete_optimization.generic_tasks_tools.objectives.objective_computer import (
    ObjectiveComputer,
)


class CalendarPreemptedComputer(
    ObjectiveComputer[Task], Generic[Task, CumulativeResource, OtherCalendarResource]
):
    problem: CalendarPreemptiveProblem[Task, CumulativeResource, OtherCalendarResource]

    @staticmethod
    def get_objective_name() -> Objective | str:
        return Objective.NB_CAL_PREEMPTED_TASKS

    def __init__(
        self,
        problem: CalendarPreemptiveProblem[
            Task, CumulativeResource, OtherCalendarResource
        ],
        weight_objective: float = 1.0,
    ):
        super().__init__(problem, weight_objective)

    def compute_objective(
        self,
        solution: CalendarPreemptiveSolution[
            Task, CumulativeResource, OtherCalendarResource
        ],
    ) -> float:
        nb_cal_preemptive = 0
        for task in self.problem.tasks_list:
            if not solution.is_present(task):
                continue
            dur = solution.get_end_time(task=task) - solution.get_start_time(task=task)
            mode = solution.get_mode(task=task)
            nominal_duration = self.problem.get_task_mode_duration(task=task, mode=mode)
            if dur != nominal_duration:
                nb_cal_preemptive += 1
        return nb_cal_preemptive
