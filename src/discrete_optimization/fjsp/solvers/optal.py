#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from collections import defaultdict
from typing import Any, Optional

from discrete_optimization.fjsp.problem import FJobShopProblem, FJobShopSolution, Task
from discrete_optimization.generic_tasks_tools.solvers.optalcp_tasks_solver import (
    SchedulingOptalSolver,
)

try:
    import optalcp as cp
except ImportError:
    cp = None

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)


class OptalFJspSolver(SchedulingOptalSolver[Task]):
    problem: FJobShopProblem

    def __init__(
        self,
        problem: FJobShopProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}

    def init_model(self, **args: Any) -> None:
        self.cp_model = cp.Model()
        intervals = {}
        opt_intervals = {}
        intervals_per_machines = defaultdict(lambda: set())
        for i in range(self.problem.n_jobs):
            for k in range(len(self.problem.list_jobs[i].sub_jobs)):
                subjob_option = self.problem.list_jobs[i].sub_jobs[k]
                durations = [opt.processing_time for opt in subjob_option]
                intervals[(i, k)] = self.cp_model.interval_var(
                    start=(0, self.problem.horizon),
                    end=(0, self.problem.horizon),
                    length=(min(durations), max(durations)),
                    optional=False,
                    name=f"interval_{i}_{k}",
                )
                for opt in subjob_option:
                    opt_intervals[(i, k, opt.machine_id)] = self.cp_model.interval_var(
                        start=(0, self.problem.horizon),
                        end=(0, self.problem.horizon),
                        length=opt.processing_time,
                        optional=True,
                        name=f"interval_{i}_{k}_{opt.machine_id}",
                    )
                    intervals_per_machines[opt.machine_id].add((i, k, opt.machine_id))
                self.cp_model.alternative(
                    intervals[(i, k)],
                    [opt_intervals[(i, k, opt.machine_id)] for opt in subjob_option],
                )
                if k >= 1:
                    self.cp_model.end_before_start(
                        intervals[(i, k - 1)], intervals[(i, k)]
                    )
        for machine in intervals_per_machines:
            self.cp_model.no_overlap(
                [opt_intervals[x] for x in intervals_per_machines[machine]]
            )
        self.variables["intervals"] = intervals
        self.variables["opt_intervals"] = opt_intervals
        self.cp_model.minimize(
            self.cp_model.max(
                [
                    self.cp_model.end(self.variables["intervals"][x])
                    for x in self.variables["intervals"]
                ]
            )
        )

    def get_task_interval_variable(self, task: Task) -> cp.IntervalVar:
        return self.variables["intervals"][task]

    def retrieve_solution(self, result: cp.SolveResult) -> Solution:
        schedule = []
        for i in range(self.problem.n_jobs):
            sched_i = []
            for k in range(len(self.problem.list_jobs[i].sub_jobs)):
                for index_opt, opt in enumerate(self.problem.list_jobs[i].sub_jobs[k]):
                    if result.solution.is_present(
                        self.variables["opt_intervals"][(i, k, opt.machine_id)]
                    ):
                        st, end = result.solution.get_value(
                            self.variables["opt_intervals"][(i, k, opt.machine_id)]
                        )
                        sched_i.append((st, end, opt.machine_id, index_opt))
            schedule.append(sched_i)
        return FJobShopSolution(problem=self.problem, schedule=schedule)
