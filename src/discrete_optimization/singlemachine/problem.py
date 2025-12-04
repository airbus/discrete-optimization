#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import List, Optional

import numpy as np

from discrete_optimization.generic_tasks_tools.scheduling import (
    SchedulingProblem,
    SchedulingSolution,
)
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveDoc,
    ObjectiveHandling,
    ObjectiveRegister,
    Problem,
    Solution,
    TypeObjective,
)
from discrete_optimization.generic_tools.encoding_register import (
    EncodingRegister,
    Permutation,
)

Task = int


class WTSolution(SchedulingSolution[Task]):
    problem: WeightedTardinessProblem

    def __init__(
        self,
        problem: WeightedTardinessProblem,
        schedule: list[tuple[int, int]] = None,
        permutation: list[int] = None,
    ):
        super().__init__(problem=problem)
        self.schedule = schedule
        self.permutation = permutation
        self.compute_schedule_from_permutation()

    def compute_schedule_from_permutation(self):
        if self.schedule is None:
            assert self.permutation is not None
            current_time = 0
            schedule = [None for i in range(self.problem.num_jobs)]
            for j in self.permutation:
                schedule[j] = (
                    current_time,
                    current_time + self.problem.processing_times[j],
                )
                current_time = schedule[j][1]
            self.schedule = schedule

    def lazy_copy(self) -> "Solution":
        return WTSolution(
            problem=self.problem,
            # schedule=self.schedule,
            permutation=self.permutation,
        )

    def copy(self) -> "Solution":
        return self.lazy_copy()

    def get_end_time(self, task: Task) -> int:
        return self.schedule[task][1]

    def get_start_time(self, task: Task) -> int:
        return self.schedule[task][0]

    def change_problem(self, new_problem: Problem) -> None:
        super().change_problem(new_problem)
        # invalidate evaluation results
        self.schedule = None
        self.compute_schedule_from_permutation()


class WeightedTardinessProblem(SchedulingProblem[Task]):
    """
    Represents a single instance of the single-machine weighted tardiness problem.
    """

    def __init__(
        self,
        num_jobs: int,
        processing_times: List[int],
        weights: List[int],
        due_dates: List[int],
        release_dates: Optional[List[int]] = None,
    ):
        if not (len(processing_times) == len(weights) == len(due_dates) == num_jobs):
            raise ValueError(f"All lists must contain {num_jobs} elements.")
        self.num_jobs = num_jobs
        self.processing_times = processing_times
        self.weights = weights
        self.due_dates = due_dates
        self.release_dates = release_dates
        if self.release_dates is None:
            self.has_release = False
            self.release_dates = [0 for i in range(self.num_jobs)]
            # We still put some dummy values in case we use models considering release...
        else:
            self.has_release = True

    @property
    def tasks_list(self) -> list[Task]:
        return list(range(self.num_jobs))

    def __repr__(self):
        return (
            f"WeightedTardinessProblem(num_jobs={self.num_jobs}, "
            f"processing_times=..., weights=..., due_dates=...)"
        )

    def get_makespan_upper_bound(self) -> int:
        return sum(self.processing_times)

    def evaluate(self, variable: WTSolution) -> dict[str, float]:
        return {
            "tardiness": sum(
                [
                    self.weights[i]
                    * max(variable.schedule[i][-1] - self.due_dates[i], 0)
                    for i in range(self.num_jobs)
                ]
            ),
            "penalty": sum(
                [
                    max(self.release_dates[i] - variable.schedule[i][0], 0)
                    for i in range(self.num_jobs)
                ]
            ),
        }

    def satisfy(self, variable: WTSolution) -> bool:
        max_t = max([x[1] for x in variable.schedule])
        active_machine = np.zeros((max_t + 1))
        for i in range(self.num_jobs):
            active_machine[variable.schedule[i][0] : variable.schedule[i][1]] += 1
            if np.max(active_machine) > 1:
                return False
            if variable.schedule[i][0] < self.release_dates[i]:
                return False
        return True

    def get_attribute_register(self) -> EncodingRegister:
        return EncodingRegister(
            {"permutation": Permutation(range=range(self.num_jobs))}
        )

    def get_solution_type(self) -> type[Solution]:
        return WTSolution

    def get_dummy_solution(self) -> WTSolution:
        return WTSolution(problem=self, permutation=list(range(self.num_jobs)))

    def get_objective_register(self) -> ObjectiveRegister:
        return ObjectiveRegister(
            objective_sense=ModeOptim.MINIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc={
                "tardiness": ObjectiveDoc(
                    type=TypeObjective.OBJECTIVE, default_weight=1
                ),
                "penalty": ObjectiveDoc(
                    type=TypeObjective.PENALTY, default_weight=10000
                ),
            },
        )
