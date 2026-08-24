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


class EarlinessTardinessComputer(ObjectiveComputer[Task]):
    problem: SchedulingProblem[Task]

    @staticmethod
    def get_objective_name() -> Objective | str:
        return Objective.EARLINESS_TARDINESS

    def __init__(
        self,
        problem: SchedulingProblem[Task],
        weight_objective: float = 1.0,
        max_start_and_weight_for_tardiness: dict[
            Task, tuple[int | None, int | None]
        ] = None,
        max_end_and_weight_for_tardiness: dict[
            Task, tuple[int | None, int | None]
        ] = None,
        min_start_and_weight_for_earliness: dict[
            Task, tuple[int | None, int | None]
        ] = None,
        min_end_and_weight_for_earliness: dict[
            Task, tuple[int | None, int | None]
        ] = None,
    ):
        super().__init__(problem, weight_objective)
        if max_start_and_weight_for_tardiness is None:
            self.max_start_and_weight_for_tardiness = {}
        else:
            self.max_start_and_weight_for_tardiness = max_start_and_weight_for_tardiness
        if max_end_and_weight_for_tardiness is None:
            self.max_end_and_weight_for_tardiness = {}
        else:
            self.max_end_and_weight_for_tardiness = max_end_and_weight_for_tardiness
        if min_start_and_weight_for_earliness is None:
            self.min_start_and_weight_for_earliness = {}
        else:
            self.min_start_and_weight_for_earliness = min_start_and_weight_for_earliness
        if min_end_and_weight_for_earliness is None:
            self.min_end_and_weight_for_earliness = {}
        else:
            self.min_end_and_weight_for_earliness = min_end_and_weight_for_earliness

    def get_max_start_for_tardiness(self, task: Task) -> int | None:
        return self.max_start_and_weight_for_tardiness.get(task, (None, 0))[0]

    def get_max_end_for_tardiness(self, task: Task) -> int:
        return self.max_end_and_weight_for_tardiness.get(task, (None, 0))[0]

    def get_min_start_for_earliness(self, task: Task) -> int:
        return self.min_start_and_weight_for_earliness.get(task, (None, 0))[0]

    def get_min_end_for_earliness(self, task: Task) -> int:
        return self.min_end_and_weight_for_earliness.get(task, (None, 0))[0]

    def get_weight_start_for_tardiness(self, task: Task) -> float:
        return self.max_start_and_weight_for_tardiness.get(task, (0, None))[1]

    def get_weight_end_for_tardiness(self, task: Task) -> float:
        return self.max_end_and_weight_for_tardiness.get(task, (0, None))[1]

    def get_weight_start_for_earliness(self, task: Task) -> float:
        return self.min_start_and_weight_for_earliness.get(task, (0, None))[1]

    def get_weight_end_for_earliness(self, task: Task) -> float:
        return self.min_end_and_weight_for_earliness.get(task, (0, None))[1]

    def get_tasks_having_max_start_for_tardiness(self) -> list[Task]:
        return [
            t
            for t in self.max_start_and_weight_for_tardiness
            if self.get_max_start_for_tardiness(t) is not None
            and self.get_weight_start_for_tardiness(t) > 0
        ]

    def get_tasks_having_max_end_for_tardiness(self) -> list[Task]:
        return [
            t
            for t in self.max_end_and_weight_for_tardiness
            if self.get_max_end_for_tardiness(t) is not None
            and self.get_weight_end_for_tardiness(t) > 0
        ]

    def get_tasks_having_min_start_for_earliness(self) -> list[Task]:
        return [
            t
            for t in self.min_start_and_weight_for_earliness
            if self.get_min_start_for_earliness(t) is not None
            and self.get_weight_start_for_earliness(t) > 0
        ]

    def get_tasks_having_min_end_for_earliness(self) -> list[Task]:
        return [
            t
            for t in self.min_end_and_weight_for_earliness
            if self.get_min_end_for_earliness(t) is not None
            and self.get_weight_end_for_earliness(t) > 0
        ]

    def compute_objective(self, solution: TasksSolution) -> float:
        return self.compute_aggregated_cost(solution)

    def compute_earliness_end(
        self, solution: SchedulingSolution[Task], task: Task
    ) -> float:
        return max(
            0, self.get_min_end_for_earliness(task) - solution.get_end_time(task)
        )

    def compute_earliness_start(
        self, solution: SchedulingSolution[Task], task: Task
    ) -> float:
        return max(
            0, self.get_min_start_for_earliness(task) - solution.get_start_time(task)
        )

    def compute_tardiness_end(
        self, solution: SchedulingSolution[Task], task: Task
    ) -> float:
        return max(
            0, solution.get_end_time(task) - self.get_max_end_for_tardiness(task)
        )

    def compute_tardiness_start(
        self, solution: SchedulingSolution[Task], task: Task
    ) -> float:
        return max(
            0, solution.get_start_time(task) - self.get_max_start_for_tardiness(task)
        )

    def detailed_objectives(
        self, solution: SchedulingSolution[Task]
    ) -> dict[tuple[str, str], dict[Task, int]]:
        return {
            ("earliness", "end"): {
                t: self.compute_earliness_end(solution, t)
                for t in self.get_tasks_having_min_end_for_earliness()
            },
            ("earliness", "start"): {
                t: self.compute_earliness_start(solution, t)
                for t in self.get_tasks_having_min_start_for_earliness()
            },
            ("tardiness", "end"): {
                t: self.compute_tardiness_end(solution, t)
                for t in self.get_tasks_having_max_end_for_tardiness()
            },
            ("tardiness", "start"): {
                t: self.compute_tardiness_start(solution, t)
                for t in self.get_tasks_having_max_start_for_tardiness()
            },
        }

    def compute_earliness_cost(self, solution: SchedulingSolution[Task]) -> float:
        earliness_start = sum(
            self.compute_earliness_start(solution, t)
            * self.get_weight_start_for_earliness(t)
            for t in self.get_tasks_having_min_start_for_earliness()
        )
        earliness_end = sum(
            self.compute_earliness_end(solution, t)
            * self.get_weight_end_for_earliness(t)
            for t in self.get_tasks_having_min_end_for_earliness()
        )
        return earliness_start + earliness_end

    def compute_tardiness_cost(self, solution: SchedulingSolution[Task]) -> float:
        tardiness_start = sum(
            self.compute_tardiness_start(solution, t)
            * self.get_weight_start_for_tardiness(t)
            for t in self.get_tasks_having_max_start_for_tardiness()
        )
        tardiness_end = sum(
            self.compute_tardiness_end(solution, t)
            * self.get_weight_end_for_tardiness(t)
            for t in self.get_tasks_having_max_end_for_tardiness()
        )
        return tardiness_start + tardiness_end

    def compute_aggregated_cost(self, solution: SchedulingSolution[Task]) -> float:
        return self.compute_earliness_cost(solution) + self.compute_tardiness_cost(
            solution
        )
