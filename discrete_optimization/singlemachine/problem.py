#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import List

import numpy as np

from discrete_optimization.generic_tools.do_problem import (
    EncodingRegister,
    ModeOptim,
    ObjectiveDoc,
    ObjectiveHandling,
    ObjectiveRegister,
    ParamsObjectiveFunction,
    Problem,
    Solution,
    TypeAttribute,
    TypeObjective,
)


class WTSolution(Solution):
    def __init__(
        self, problem: "WeightedTardinessProblem", schedule: list[tuple[int, int]]
    ):
        self.problem = problem
        self.schedule = schedule


class WeightedTardinessProblem(Problem):
    """
    Represents a single instance of the single-machine weighted tardiness problem.
    """

    def __init__(
        self,
        num_jobs: int,
        processing_times: List[int],
        weights: List[int],
        due_dates: List[int],
    ):
        if not (len(processing_times) == len(weights) == len(due_dates) == num_jobs):
            raise ValueError(f"All lists must contain {num_jobs} elements.")
        self.num_jobs = num_jobs
        self.processing_times = processing_times
        self.weights = weights
        self.due_dates = due_dates

    def __repr__(self):
        return (
            f"WeightedTardinessProblem(num_jobs={self.num_jobs}, "
            f"processing_times=..., weights=..., due_dates=...)"
        )

    def evaluate(self, variable: WTSolution) -> dict[str, float]:
        return {
            "tardiness": sum(
                [
                    self.weights[i]
                    * max(variable.schedule[i][-1] - self.due_dates[i], 0)
                    for i in range(len(self.due_dates))
                ]
            )
        }

    def satisfy(self, variable: WTSolution) -> bool:
        max_t = max([x[1] for x in variable.schedule])
        active_machine = np.zeros((max_t + 1))
        for i in range(self.num_jobs):
            active_machine[variable.schedule[i][0] : variable.schedule[i][1]] += 1
            if np.max(active_machine) > 1:
                return False
        return True

    def get_attribute_register(self) -> EncodingRegister:
        return EncodingRegister(
            dict_attribute_to_type={"schedule": {"n": self.num_jobs}}
        )

    def get_solution_type(self) -> type[Solution]:
        return WTSolution

    def get_objective_register(self) -> ObjectiveRegister:
        return ObjectiveRegister(
            objective_sense=ModeOptim.MINIMIZATION,
            objective_handling=ObjectiveHandling.SINGLE,
            dict_objective_to_doc={
                "tardiness": ObjectiveDoc(
                    type=TypeObjective.OBJECTIVE, default_weight=1
                )
            },
        )

    def get_dummy_solution(self) -> WTSolution:
        schedule = []
        time_ = 0
        for i in range(self.num_jobs):
            schedule.append((time_, time_ + self.processing_times[i]))
            time_ = schedule[-1][1]
        return WTSolution(problem=self, schedule=schedule)
