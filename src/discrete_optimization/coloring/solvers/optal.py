#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from typing import Any, Optional

try:
    import optalcp as cp
except ImportError:
    cp = None

from discrete_optimization.coloring.problem import (
    Color,
    ColoringProblem,
    ColoringSolution,
    Node,
)
from discrete_optimization.coloring.utils import compute_cliques
from discrete_optimization.generic_tasks_tools.allocation import UnaryResource
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.generic_tasks_tools.solvers.optalcp_tasks_solver import (
    AllocationOptalSolver,
    SchedulingOptalSolver,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    IntegerHyperparameter,
)


class ModelingCpSat(Enum):
    BINARY = 0
    INTEGER = 1


class OptalColoringSolver(
    AllocationOptalSolver[Node, Color], SchedulingOptalSolver[Node]
):
    hyperparameters = [
        CategoricalHyperparameter("use_clique", choices=[True, False], default=False),
        IntegerHyperparameter(
            "max_cliques",
            low=0,
            high=10000,
            default=1000,
            depends_on=[("use_clique", True)],
        ),
    ]

    def get_task_unary_resource_is_present_variable(
        self, task: Task, unary_resource: UnaryResource
    ) -> cp.BoolExpr:
        return self.get_task_interval_variable(
            task
        ) == self.problem.get_index_from_unary_resource(unary_resource)

    def get_task_interval_variable(self, task: Task) -> cp.IntervalVar:
        return self.variables["intervals"][task]

    def retrieve_solution(self, result: cp.SolveResult) -> Solution:
        color = [
            result.solution.get_start(self.get_task_interval_variable(t))
            for t in self.problem.tasks_list
        ]
        return ColoringSolution(problem=self.problem, colors=color)

    at_most_one_unary_resource_per_task = True
    problem: ColoringProblem
    _nb_colors: int

    def __init__(
        self,
        problem: ColoringProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}
        self._subset_nodes = list(self.problem.subset_nodes)

    def init_model(self, **args: Any) -> None:
        args = self.complete_with_default_hyperparameters(args)
        self.cp_model = cp.Model()
        args["nb_colors"] = min(args["nb_colors"], self.problem.number_of_nodes)
        self._nb_colors = args["nb_colors"]
        nb_colors = args["nb_colors"]
        intervals = {}
        for n in self.problem.tasks_list:
            intervals[n] = self.cp_model.interval_var(
                start=(0, nb_colors - 1),
                end=(1, nb_colors - 1),
                length=1,
                optional=False,
                name=f"interval_{n}",
            )
        for i, j, _ in self.problem.graph.edges:
            self.cp_model.enforce(
                self.cp_model.start(intervals[i]) != self.cp_model.start(intervals[j])
            )
        if args["use_clique"]:
            g = self.problem.graph.to_networkx()
            cliques, not_all = compute_cliques(g, args["max_cliques"])
            for clique in cliques:
                print(len(clique))
                self.cp_model.no_overlap([intervals[c] for c in clique])
        self.cp_model.minimize(
            self.cp_model.max(
                [self.cp_model.end(intervals[i]) for i in self.problem.subset_nodes]
            )
        )
        self.variables["intervals"] = intervals

    @property
    def subset_tasks_of_interest(self) -> Iterable[Node]:
        return self.problem.subset_nodes

    @property
    def subset_unaryresources_allowed(self) -> Iterable[Color]:
        return range(self._nb_colors)

    def create_used_variables_list(
        self,
    ) -> list["cp.BoolVar"]:
        self.create_used_variables()
        return [
            self.used_variables[color] for color in self.subset_unaryresources_allowed
        ]
