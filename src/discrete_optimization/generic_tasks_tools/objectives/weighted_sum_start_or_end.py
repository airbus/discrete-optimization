#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.generic_tasks_tools.entities import SchedulingEntity
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import Objective
from discrete_optimization.generic_tasks_tools.objectives.objective_computer import (
    ObjectiveComputer,
)
from discrete_optimization.generic_tasks_tools.scheduling import (
    SchedulingProblem,
    SchedulingSolution,
    Task,
)


class WeightedSumStartEndObjectiveComputer(ObjectiveComputer[Task]):
    problem: SchedulingProblem[Task]

    def __init__(
        self,
        problem: SchedulingProblem[Task],
        dict_weight: dict[Task | SchedulingEntity[Task], dict[StartOrEnd, int]] = None,
        weight_objective: float = 1.0,
    ):
        super().__init__(problem, weight_objective=weight_objective)
        self.dict_weight = dict_weight
        if dict_weight is None:
            # by default, sum of end time.
            self.dict_weight = {t: {StartOrEnd.END: 1} for t in self.problem.tasks_list}

    @staticmethod
    def get_objective_name() -> Objective | str:
        return Objective.WEIGHTED_SUM_START_OR_END

    def compute_objective(self, solution: SchedulingSolution[Task]) -> int:
        v = 0
        for t in self.dict_weight:
            if isinstance(t, SchedulingEntity):
                if not t.is_active(solution):
                    continue
                for s in self.dict_weight[t]:
                    if s == StartOrEnd.START:
                        v += self.dict_weight[t][s] * t.get_start_time(solution)
                    if s == StartOrEnd.END:
                        v += self.dict_weight[t][s] * t.get_end_time(solution)
            else:
                if not solution.is_present(t):
                    continue
                for s in self.dict_weight[t]:
                    if s == StartOrEnd.START:
                        v += self.dict_weight[t][s] * solution.get_start_time(t)
                    if s == StartOrEnd.END:
                        v += self.dict_weight[t][s] * solution.get_end_time(t)
        return v
