#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from typing import Any

from discrete_optimization.fjsp.problem import (
    CumulativeResource,
    FJobShopProblem,
    FJobShopSolution,
    Task,
)
from discrete_optimization.generic_tasks_tools.allocation import (
    NoUnaryResource,
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.non_renewable_resource import (
    NoNonRenewableResource,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.auto import (
    GenericSchedulingAutoCpSatSolver,
    TemporarySolution,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
)

logger = logging.getLogger(__name__)


class CpSatAutoFjspSolver(
    GenericSchedulingAutoCpSatSolver[
        Task, NoUnaryResource, CumulativeResource, NoNonRenewableResource
    ],
):
    hyperparameters = [
        CategoricalHyperparameter(
            name="duplicate_temporal_var", choices=[True, False], default=False
        ),
        CategoricalHyperparameter(
            name="add_cumulative_constraint", choices=[True, False], default=False
        ),
    ]
    problem: FJobShopProblem

    def init_model(self, **kwargs: Any) -> None:
        # optional parameters
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        self._max_time = kwargs.get(
            "max_time", self.problem.get_makespan_upper_bound()
        )  # update the upper bound for makespan
        self.duplicate_start_var_per_mode = kwargs["duplicate_temporal_var"]
        # compute start/end of task lower bounds by forward propagation
        self.compute_start_and_end_lower_bound()
        # whether to add cumulative constraint on top of no_overlap constraint
        self.add_no_overlap_and_cumulative = bool(kwargs["add_cumulative_constraint"])
        super().init_model(**kwargs)

    def get_makespan_upper_bound(self) -> int:
        return self._max_time

    def compute_start_and_end_lower_bound(self) -> None:
        self.start_lower_bound: dict[Task, int] = {}
        self.end_lower_bound: dict[Task, int] = {}

        for j, job in enumerate(self.problem.list_jobs):
            lb = 0
            for k, sub_job_options in enumerate(job.sub_jobs):
                task = j, k
                possible_durations = [
                    sub_job.processing_time for sub_job in sub_job_options
                ]
                self.start_lower_bound[task] = lb
                lb += min(possible_durations)
                self.end_lower_bound[task] = lb

    def get_task_start_or_end_lower_bound(
        self, task: Task, start_or_end: StartOrEnd
    ) -> int:
        match start_or_end:
            case StartOrEnd.START:
                return self.start_lower_bound[task]
            case _:
                return self.end_lower_bound[task]

    def convert_task_variables_to_solution(
        self, temp_sol: TemporarySolution[Task, UnaryResource]
    ) -> FJobShopSolution:
        schedule = [
            [
                (
                    (task_var := temp_sol.task_variables[j, k]).start,
                    task_var.end,
                    self.problem.mode2machine[j, k][task_var.mode],
                    task_var.mode,
                )
                for k, sub_job in enumerate(job.sub_jobs)
            ]
            for j, job in enumerate(self.problem.list_jobs)
        ]
        return FJobShopSolution(problem=self.problem, schedule=schedule)
