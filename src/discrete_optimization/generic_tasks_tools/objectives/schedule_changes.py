#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tasks_tools.base import TasksSolution
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import Objective
from discrete_optimization.generic_tasks_tools.objectives.objective_computer import (
    ObjectiveComputer,
)
from discrete_optimization.generic_tasks_tools.scheduling import (
    SchedulingProblem,
    SchedulingSolution,
    Task,
)


class ScheduleChangesComputer(ObjectiveComputer[Task]):
    base_scheduling_solution: SchedulingSolution[Task]
    problem: SchedulingProblem[Task]

    @staticmethod
    def get_objective_name() -> Objective | str:
        return Objective.SCHEDULE_CHANGES

    def __init__(
        self,
        problem: SchedulingProblem[Task],
        base_scheduling_solution: SchedulingSolution[Task],
        weight_objective: float = 1.0,
        cost_any_shift: dict[Task, float] = None,
        cost_unit_deviation: dict[Task, float] = None,
    ):
        super().__init__(problem, weight_objective)
        self.base_scheduling_solution = base_scheduling_solution
        if cost_any_shift is None:
            self._cost_any_shift = {}
        else:
            self._cost_any_shift = cost_any_shift
        if cost_unit_deviation is None:
            self._cost_unit_deviation = {}
        else:
            self._cost_unit_deviation = cost_unit_deviation

    def cost_any_shift(self, task: Task) -> int:
        """
        Cost of any move of the task (whatever deviation)
        :param task: task of the scheduling problem
        :return: the cost of any move of the task
        """
        return self._cost_any_shift.get(task, 0)

    def cost_unit_deviation(self, task: Task) -> int:
        """
        Cost of shifting one unit of time
        so the total cost will be : unit*abs(new_time-prev_time)
        """
        return self._cost_unit_deviation.get(task, 0)

    def tasks_with_any_shift_cost(self):
        return [t for t in self._cost_any_shift if self.cost_any_shift(t) > 0]

    def tasks_with_unit_deviation_cost(self):
        return [t for t in self._cost_unit_deviation if self.cost_unit_deviation(t) > 0]

    def has_any_shift_cost(self):
        return len(self.tasks_with_any_shift_cost()) > 0

    def has_any_unit_deviation_cost(self):
        return len(self.tasks_with_unit_deviation_cost()) > 0

    def compute_any_shift_cost(self, solution: SchedulingSolution[Task]) -> float:
        if not self.has_any_shift_cost():
            return 0
        return sum(
            [
                self.cost_any_shift(task)
                for task in self.tasks_with_any_shift_cost()
                if self.base_scheduling_solution.get_start_time(task)
                != solution.get_start_time(task)
            ]
        )

    def compute_unit_deviation_cost(self, solution: SchedulingSolution[Task]) -> float:
        if not self.has_any_unit_deviation_cost():
            return 0
        return sum(
            [
                self.cost_unit_deviation(task)
                * abs(
                    solution.get_start_time(task)
                    - self.base_scheduling_solution.get_start_time(task)
                )
                for task in self.tasks_with_unit_deviation_cost()
            ]
        )

    def compute_objective(self, solution: TasksSolution) -> float:
        return self.compute_any_shift_cost(solution) + self.compute_unit_deviation_cost(
            solution
        )
