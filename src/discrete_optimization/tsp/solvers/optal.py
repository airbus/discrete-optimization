#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from typing import Any

import optalcp as cp

from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.solvers.optalcp_tasks_solver import (
    SchedulingOptalSolver,
)
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.tsp.problem import Node, TspProblem, TspSolution
from discrete_optimization.tsp.utils import build_matrice_distance

logger = logging.getLogger(__name__)


class OptalTspSolver(SchedulingOptalSolver[Node]):
    problem: TspProblem

    def __init__(
        self,
        problem: TspProblem,
        params_objective_function: ParamsObjectiveFunction = None,
        **kwargs,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.distance_matrix = build_matrice_distance(
            self.problem.node_count,
            method=self.problem.evaluate_function_indexes,
        )
        self.distance_matrix[self.problem.end_index, self.problem.start_index] = 0
        self.variables = {}

    def init_model(self, **args: Any) -> None:
        self.cp_model = cp.Model()
        upper_bound = int(sum(self.distance_matrix.max(axis=1)))
        visits = [
            self.cp_model.interval_var(
                start=(0, None),
                end=(0, upper_bound),
                length=0,
                optional=False,
                name=f"visit_{i}",
            )
            for i in range(self.problem.node_count)
        ]
        self.variables["visits"] = visits
        seq = self.cp_model.sequence_var(visits)
        self.cp_model.enforce(
            self.cp_model.start(visits[self.problem.start_index]) == 0
        )
        seq.no_overlap()
        self.cp_model.no_overlap(
            seq,
            [
                [
                    int(self.distance_matrix[i][j])
                    for j in range(self.problem.node_count)
                ]
                for i in range(self.problem.node_count)
            ],
        )
        if self.problem.start_index == self.problem.end_index:
            come_back_base = self.cp_model.interval_var(
                start=(0, None),
                end=(0, upper_bound),
                length=0,
                optional=False,
                name=f"come_back_base",
            )
            self.variables["come_back_base"] = come_back_base
            for i in range(self.problem.node_count):
                self.cp_model.end_before_start(
                    visits[i],
                    come_back_base,
                    int(self.distance_matrix[i, self.problem.start_index]),
                )
            self.cp_model.minimize(self.cp_model.end(come_back_base))
        else:
            for i in range(self.problem.node_count):
                if i != self.problem.end_index:
                    self.cp_model.end_before_start(
                        visits[i], visits[self.problem.end_index]
                    )
            self.cp_model.minimize(self.cp_model.end(visits[self.problem.end_index]))

    def get_task_interval_variable(self, task: Task) -> cp.IntervalVar:
        return self.variables["visits"][task]

    def retrieve_solution(self, result: cp.SolveResult) -> TspSolution:
        logger.info(f"Current obj {result.solution.get_objective()}")
        starts = [
            result.solution.get_start(self.variables["visits"][i])
            for i in range(self.problem.node_count)
        ]
        ordered = sorted(range(len(starts)), key=lambda i: starts[i])
        solution = TspSolution(
            problem=self.problem,
            start_index=self.problem.start_index,
            end_index=self.problem.end_index,
            permutation=[
                o
                for o in ordered
                if o not in {self.problem.start_index, self.problem.end_index}
            ],
        )
        eval_ = self.problem.evaluate(solution)
        logger.info(f"{eval_}")
        return solution
