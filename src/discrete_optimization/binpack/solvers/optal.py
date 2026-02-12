#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Any

try:
    import optalcp as cp
except ImportError:
    cp = None
from discrete_optimization.binpack.problem import (
    BinPack,
    BinPackProblem,
    BinPackSolution,
    Item,
)
from discrete_optimization.generic_tasks_tools.allocation import UnaryResource
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.solvers.optalcp_tasks_solver import (
    AllocationOptalSolver,
    SchedulingOptalSolver,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)


class OptalBinPackSolver(
    AllocationOptalSolver[Item, BinPack],
    SchedulingOptalSolver[Item],
):
    problem: BinPackProblem

    def __init__(
        self,
        problem: BinPackProblem,
        params_objective_function: ParamsObjectiveFunction | None = None,
        **kwargs,
    ) -> None:
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}

    def init_model(self, **kwargs: Any) -> None:
        self.cp_model = cp.Model()
        upper_bound = min(
            kwargs.get("upper_bound", self.problem.nb_items), self.problem.nb_items
        )
        intervals = {}
        for t in self.problem.tasks_list:
            intervals[t] = self.cp_model.interval_var(
                start=(0, upper_bound - 1),
                end=(1, upper_bound),
                length=1,
                name=f"interval_item_{t}",
            )
        if self.problem.has_constraint:
            for i, j in self.problem.incompatible_items:
                self.cp_model.enforce(
                    self.cp_model.start(intervals[i])
                    != self.cp_model.start(intervals[j])
                )
        capacity = self.problem.capacity_bin
        self.cp_model.enforce(
            self.cp_model.sum(
                [
                    self.cp_model.pulse(intervals[t], self.problem.list_items[t].weight)
                    for t in intervals
                ]
            )
            <= capacity
        )
        self.variables["intervals"] = intervals
        self.cp_model.minimize(
            self.cp_model.max([self.cp_model.end(intervals[t]) for t in intervals])
        )

    def get_task_unary_resource_is_present_variable(
        self, task: Task, unary_resource: UnaryResource
    ) -> "cp.BoolExpr":
        index = self.problem.get_index_from_unary_resource(unary_resource)
        return (
            self.get_task_start_or_end_variable(task, start_or_end=StartOrEnd.START)
            == index
        )

    def get_task_interval_variable(self, task: Task) -> "cp.IntervalVar":
        return self.variables["intervals"][task]

    def retrieve_solution(self, result: cp.SolveResult) -> Solution:
        allocation = [
            int(result.solution.get_start(self.get_task_interval_variable(t)))
            for t in self.problem.tasks_list
        ]
        return BinPackSolution(problem=self.problem, allocation=allocation)
