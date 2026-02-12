#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any

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
from discrete_optimization.salbp.problem import SalbpProblem, SalbpSolution, Task


class OptalSalbpSolver(SchedulingOptalSolver[Task]):
    problem: SalbpProblem

    def __init__(
        self,
        problem: SalbpProblem,
        params_objective_function: ParamsObjectiveFunction | None = None,
        **kwargs,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}

    def init_model(self, **kwargs: Any) -> None:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        self.init_model_scheduling(**kwargs)

    def init_model_scheduling(self, **kwargs: Any) -> None:
        self.cp_model = cp.Model()
        upper_bound = kwargs.get("upper_bound", self.problem.get_makespan_upper_bound())
        intervals = {}
        for t in self.problem.tasks:
            intervals[t] = self.cp_model.interval_var(
                start=(0, upper_bound),
                end=(0, upper_bound),
                length=1,
                name=f"intervals_{t}",
            )
        for t in self.problem.adj:
            for succ in self.problem.adj[t]:
                self.cp_model.start_before_start(intervals[t], intervals[succ])
        self.cp_model.enforce(
            self.cp_model.sum(
                [
                    self.cp_model.pulse(intervals[t], self.problem.task_times[t])
                    for t in self.problem.tasks
                ]
            )
            <= self.problem.cycle_time
        )
        self.variables["intervals"] = intervals
        makespan = self.get_global_makespan_variable()
        self.cp_model.minimize(makespan)

    def retrieve_solution(self, result: "cp.SolveResult") -> Solution:
        allocation = [
            int(result.solution.get_start(self.get_task_interval_variable(task)))
            for task in self.problem.tasks
        ]
        return SalbpSolution(problem=self.problem, allocation_to_station=allocation)

    def get_task_interval_variable(self, task: Task) -> "cp.IntervalVar":
        return self.variables["intervals"][task]
