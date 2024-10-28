#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any

from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.toulbar_tools import ToulbarSolver
from discrete_optimization.knapsack.problem import KnapsackSolution
from discrete_optimization.knapsack.solvers.knapsack_solver import KnapsackSolver

try:
    import pytoulbar2

    toulbar_available = True
except ImportError as e:
    toulbar_available = False


class ToulbarKnapsackSolver(ToulbarSolver, KnapsackSolver, WarmstartMixin):
    def init_model(self, **kwargs: Any) -> None:
        model = pytoulbar2.CFN(vns=kwargs.get("vns", None))
        for i in range(self.problem.nb_items):
            model.AddVariable(f"x_{i}", values=[0, 1])
            model.AddFunction([f"x_{i}"], [0, -self.problem.list_items[i].value])
        model.AddLinearConstraint(
            [self.problem.list_items[i].weight for i in range(self.problem.nb_items)],
            [f"x_{i}" for i in range(self.problem.nb_items)],
            operand="<=",
            rightcoef=self.problem.max_capacity,
        )
        self.model = model

    def retrieve_solution(
        self, solution_from_toulbar2: tuple[list, float, int]
    ) -> Solution:
        return KnapsackSolution(
            problem=self.problem, list_taken=solution_from_toulbar2[0]
        )

    def set_warm_start(self, solution: KnapsackSolution) -> None:
        for i in range(len(solution.list_taken)):
            self.model.CFN.wcsp.setBestValue(i, solution.list_taken[i])
