#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  JSP solver using OptalCp solver python api.
from typing import Any, Optional

import optalcp as cp

from discrete_optimization.generic_tasks_tools.solvers.optalcp_tasks_solver import (
    SchedulingOptalSolver,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.jsp.problem import JobShopProblem, JobShopSolution, Task


class OptalJspSolver(SchedulingOptalSolver[Task]):
    problem: JobShopProblem

    def __init__(
        self,
        problem: JobShopProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}

    def init_model(self, **args: Any) -> None:
        self.cp_model = cp.Model()
        intervals = {}
        for job_index in range(self.problem.n_jobs):
            for k in range(len(self.problem.list_jobs[job_index])):
                processing = self.problem.list_jobs[job_index][k].processing_time
                intervals[(job_index, k)] = self.cp_model.interval_var(
                    start=(0, None),
                    end=(processing, None),
                    length=processing,
                    name=f"job_{job_index}_{k}",
                )
            for k in range(1, len(self.problem.list_jobs[job_index])):
                self.cp_model.end_before_start(
                    intervals[(job_index, k - 1)], intervals[(job_index, k)]
                )
        for m in self.problem.job_per_machines:
            self.cp_model.no_overlap(
                [intervals[x] for x in self.problem.job_per_machines[m]]
            )
        self.variables["intervals"] = intervals
        self.cp_model.minimize(
            self.cp_model.max(
                [
                    self.cp_model.end(
                        intervals[(job, len(self.problem.list_jobs[job]) - 1)]
                    )
                    for job in range(self.problem.n_jobs)
                ]
            )
        )

    def get_task_interval_variable(self, task: Task) -> cp.IntervalVar:
        return self.variables["intervals"][task]

    def retrieve_solution(self, result: cp.SolveResult) -> Solution:
        schedule = []
        for i in range(self.problem.n_jobs):
            sched_i = []
            for k in range(len(self.problem.list_jobs[i])):
                sched_i.append(
                    result.solution.get_value(self.variables["intervals"][(i, k)])
                )
            schedule.append(sched_i)
        return JobShopSolution(problem=self.problem, schedule=schedule)
