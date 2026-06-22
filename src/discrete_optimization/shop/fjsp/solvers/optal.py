#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

from discrete_optimization.generic_tasks_tools.solvers.optalcp.scheduling import (
    SchedulingOptalSolver,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.shop.fjsp.problem import (
    FJobShopProblem,
    FJobShopSolution,
    Task,
)

try:
    import optalcp as cp
except ImportError:
    cp = None
    optalcp_available = False
else:
    optalcp_available = True


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
            for k in range(len(self.problem.list_jobs[i].subjobs)):
                subjob_recipes = self.problem.list_jobs[i].subjobs[k].recipes
                durations = [opt.processing_time for opt in subjob_recipes]
                intervals[(i, k)] = self.cp_model.interval_var(
                    start=(0, self.problem.horizon),
                    end=(0, self.problem.horizon),
                    length=(min(durations), max(durations)),
                    optional=False,
                    name=f"interval_{i}_{k}",
                )
                for opt in subjob_recipes:
                    opt_intervals[(i, k, opt.machine_index)] = (
                        self.cp_model.interval_var(
                            start=(0, self.problem.horizon),
                            end=(0, self.problem.horizon),
                            length=opt.processing_time,
                            optional=True,
                            name=f"interval_{i}_{k}_{opt.machine_index}",
                        )
                    )
                    intervals_per_machines[opt.machine_index].add(
                        (i, k, opt.machine_index)
                    )
                self.cp_model.alternative(
                    intervals[(i, k)],
                    [
                        opt_intervals[(i, k, opt.machine_index)]
                        for opt in subjob_recipes
                    ],
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
        machine_index = []
        recipe_index = []
        for i in range(self.problem.n_jobs):
            sched_i = []
            machine_index_i = []
            recipe_index_i = []
            for k in range(len(self.problem.list_jobs[i].subjobs)):
                for index_opt, opt in enumerate(
                    self.problem.list_jobs[i].subjobs[k].recipes
                ):
                    if result.solution.is_present(
                        self.variables["opt_intervals"][(i, k, opt.machine_index)]
                    ):
                        st, end = result.solution.get_value(
                            self.variables["opt_intervals"][(i, k, opt.machine_index)]
                        )
                        sched_i.append((st, end))
                        machine_index_i.append(opt.machine_index)
                        recipe_index_i.append(index_opt)
            schedule.append(sched_i)
            machine_index.append(machine_index_i)
            recipe_index.append(recipe_index_i)
        return FJobShopSolution(
            problem=self.problem,
            schedule=schedule,
            machine_index=machine_index,
            recipe_index=recipe_index,
        )
