#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

import logging
from typing import Optional

try:
    import optalcp as cp
except ImportError:
    cp = None
from discrete_optimization.generic_tasks_tools.allocation import UnaryResource
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.solvers.optalcp_tasks_solver import (
    AllocationOptalSolver,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.knapsack.problem import (
    Item,
    Knapsack,
    KnapsackProblem,
    KnapsackSolution,
)
from discrete_optimization.knapsack.solvers import KnapsackSolver

logger = logging.getLogger(__name__)


class OptalKnapsackSolver(AllocationOptalSolver[Item, Knapsack], KnapsackSolver):
    problem: KnapsackProblem

    def __init__(
        self,
        problem: KnapsackProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.variables = {}

    def init_model(self, **kwargs) -> None:
        self.cp_model = cp.Model()
        intervals = {}
        for i in self.problem.tasks_list:
            intervals[i] = self.cp_model.interval_var(
                start=0, end=1, length=1, optional=True, name=f"taken_{i}"
            )
        self.cp_model.enforce(
            self.cp_model.sum(
                [self.cp_model.pulse(intervals[i], i.weight) for i in intervals]
            )
            <= self.problem.max_capacity
        )
        self.cp_model.maximize(
            self.cp_model.sum(
                [self.cp_model.presence(intervals[i]) * i.value for i in intervals]
            )
        )
        self.variables["intervals"] = intervals

    def get_task_unary_resource_is_present_variable(
        self, task: Task, unary_resource: UnaryResource
    ) -> cp.BoolExpr:
        return self.cp_model.presence(self.variables["intervals"])

    def retrieve_solution(self, result: cp.SolveResult) -> Solution:
        taken = [
            result.solution.is_present(self.variables["intervals"][t])
            for t in self.problem.tasks_list
        ]
        return KnapsackSolution(problem=self.problem, list_taken=taken)
