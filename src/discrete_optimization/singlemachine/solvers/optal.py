#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any, Optional

import optalcp as cp

from discrete_optimization.generic_tasks_tools.solvers.optalcp_tasks_solver import (
    SchedulingOptalSolver,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
)
from discrete_optimization.singlemachine.problem import (
    Task,
    WeightedTardinessProblem,
    WTSolution,
)


class OptalSingleMachineSolver(SchedulingOptalSolver[Task]):
    problem: WeightedTardinessProblem

    def __init__(
        self,
        problem: WeightedTardinessProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}

    def init_model(self, **args: Any) -> None:
        self.cp_model = cp.Model()
        intervals = {}
        for i in self.problem.tasks_list:
            intervals[i] = self.cp_model.interval_var(
                start=(self.problem.release_dates[i], None),
                end=(
                    self.problem.release_dates[i] + self.problem.processing_times[i],
                    None,
                ),
                length=self.problem.processing_times[i],
                optional=False,
                name=f"interval-{i}",
            )
        self.cp_model.no_overlap([intervals[i] for i in intervals])
        weighted_tardiness = self.cp_model.sum(
            [
                self.cp_model.max2(
                    self.problem.weights[i]
                    * (self.cp_model.end(intervals[i]) - self.problem.due_dates[i]),
                    0,
                )
                for i in intervals
            ]
        )
        self.cp_model.minimize(weighted_tardiness)
        self.variables["intervals"] = intervals

    def get_task_interval_variable(self, task: Task) -> cp.IntervalVar:
        return self.variables["intervals"][task]

    def retrieve_solution(self, result: cp.SolveResult) -> WTSolution:
        schedule = []
        for i in self.problem.tasks_list:
            schedule.append(
                result.solution.get_value(self.get_task_interval_variable(i))
            )
        return WTSolution(problem=self.problem, schedule=schedule)
