#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  Adaptation of
#  https://github.com/erachelson/seq_dec_mak/blob/main/scheduling_newcourse/correction/nb2_jobshopsolver.py
import logging
from typing import Any

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
    SinglemodeGenericSchedulingAutoCpSatSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.skill import (
    WithoutSkillSchedulingCpSatSolver,
)
from discrete_optimization.jsp.problem import (
    JobShopProblem,
    JobShopSolution,
    NonSkillCumulativeResource,
    Task,
)

logger = logging.getLogger(__name__)


class CpSatAutoJspSolver(
    SinglemodeGenericSchedulingAutoCpSatSolver[
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
    problem: JobShopProblem

    def get_makespan_upper_bound(self) -> int:
        if self._max_time is None:
            return super().get_makespan_upper_bound()
        else:
            return min(self._max_time, super().get_makespan_upper_bound())

    def init_model(self, **kwargs: Any) -> None:
        self._max_time: int | None = kwargs.get(
            "max_time",
            None,
        )  # new horizon

        super().init_model(**kwargs)

    def convert_task_variables_to_solution(
        self, raw_sol: RawSolution[Task, UnaryResource, NoSkill]
    ) -> JobShopSolution:
        schedule = [
            [
                ((task_var := raw_sol.task_variables[j, k]).start, task_var.end)
                for k, sub_job in enumerate(job)
            ]
            for j, job in enumerate(self.problem.list_jobs)
        ]
        return JobShopSolution(problem=self.problem, schedule=schedule)
