#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from typing import Any

from discrete_optimization.fjsp.problem import (
    FJobShopProblem,
    FJobShopSolution,
    NonSkillCumulativeResource,
    Task,
)
from discrete_optimization.generic_tasks_tools.allocation import (
    NoUnaryResource,
    UnaryResource,
)
from discrete_optimization.generic_tasks_tools.generic_scheduling_utils import (
    RawSolution,
)
from discrete_optimization.generic_tasks_tools.non_renewable_resource import (
    NoNonRenewableResource,
)
from discrete_optimization.generic_tasks_tools.skill import NoSkill
from discrete_optimization.generic_tasks_tools.solvers.cpsat.auto import (
    GenericSchedulingAutoCpSatSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.skill import (
    WithoutSkillSchedulingCpSatSolver,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
)

logger = logging.getLogger(__name__)


class CpSatAutoFjspSolver(
    GenericSchedulingAutoCpSatSolver[
        Task,
        NoUnaryResource,
        NoSkill,
        NonSkillCumulativeResource,
        NoNonRenewableResource,
    ],
    WithoutSkillSchedulingCpSatSolver[
        Task, NoUnaryResource, NonSkillCumulativeResource, NoUnaryResource
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
        self._max_time: int | None = kwargs.get(
            "max_time", None
        )  # update the upper bound for makespan
        self.duplicate_start_var_per_mode = kwargs["duplicate_temporal_var"]
        # whether to add cumulative constraint on top of no_overlap constraint
        self.use_cumulative_for_capa_1 = bool(kwargs["add_cumulative_constraint"])
        # use cpm to compute start/end bounds
        self.use_cpm_for_task_bounds = True
        super().init_model(**kwargs)

    def get_makespan_upper_bound(self) -> int:
        if self._max_time is None:
            return super().get_makespan_upper_bound()
        else:
            return min(self._max_time, super().get_makespan_upper_bound())

    def convert_task_variables_to_solution(
        self, raw_sol: RawSolution[Task, UnaryResource, NoSkill]
    ) -> FJobShopSolution:
        schedule = [
            [
                (
                    (task_var := raw_sol.task_variables[j, k]).start,
                    task_var.end,
                    self.problem.mode2machine[j, k][task_var.mode],
                    task_var.mode,
                )
                for k, sub_job in enumerate(job.sub_jobs)
            ]
            for j, job in enumerate(self.problem.list_jobs)
        ]
        return FJobShopSolution(problem=self.problem, schedule=schedule)
