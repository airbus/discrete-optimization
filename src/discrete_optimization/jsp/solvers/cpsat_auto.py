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
from discrete_optimization.generic_tasks_tools.cumulative_resource import (
    NoCumulativeResource,
)
from discrete_optimization.generic_tasks_tools.non_renewable_resource import (
    NoNonRenewableResource,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat.auto import (
    SinglemodeGenericSchedulingAutoCpSatSolver,
    TemporarySolution,
)
from discrete_optimization.jsp.problem import JobShopProblem, JobShopSolution, Task

logger = logging.getLogger(__name__)


class CpSatAutoJspSolver(
    SinglemodeGenericSchedulingAutoCpSatSolver[
        Task, NoUnaryResource, NoCumulativeResource, NoNonRenewableResource
    ],
):
    problem: JobShopProblem

    def get_makespan_upper_bound(self) -> int:
        return self._max_time

    def init_model(self, **kwargs: Any) -> None:
        # dummy value, todo : compute a better bound
        max_time = kwargs.get(
            "max_time",
            self.problem.get_makespan_upper_bound(),
        )
        self._max_time = max_time  # will be used by the makespan variable

        super().init_model(**kwargs)

        # No overlap task on the same machine.
        for machine, tasks in self.problem.job_per_machines.items():
            self.cp_model.add_no_overlap(
                self.get_task_interval(task=task) for task in tasks
            )

    def convert_task_variables_to_solution(
        self, temp_sol: TemporarySolution[Task, UnaryResource]
    ) -> JobShopSolution:
        schedule = [
            [
                ((task_var := temp_sol.task_variables[j, k]).start, task_var.end)
                for k, sub_job in enumerate(job)
            ]
            for j, job in enumerate(self.problem.list_jobs)
        ]
        return JobShopSolution(problem=self.problem, schedule=schedule)
