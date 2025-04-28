#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
# Thanks to Leuven university for the cpmyp library.
from typing import Any, Optional

from cpmpy import Model, boolvar

from discrete_optimization.generic_tools.cpmpy_tools import (
    CpmpySolver,
    MetaCpmpyConstraint,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.knapsack.problem import KnapsackProblem, KnapsackSolution
from discrete_optimization.knapsack.solvers import KnapsackSolver


class CpmpyKnapsackSolver(CpmpySolver, KnapsackSolver):
    def __init__(
        self,
        problem: KnapsackProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.variables: dict[str, Any] = {}

    def init_model(self, **kwargs: Any) -> None:
        values = [
            self.problem.list_items[i].value for i in range(self.problem.nb_items)
        ]
        weights = [
            self.problem.list_items[i].weight for i in range(self.problem.nb_items)
        ]
        capacity = self.problem.max_capacity
        # Construct the model.
        x = boolvar(shape=self.problem.nb_items, name="x")
        self.model = Model(sum(x * weights) <= capacity, maximize=sum(x * values))
        self.variables["x"] = x

    def retrieve_current_solution(self) -> Solution:
        list_taken = self.variables["x"].value()
        return KnapsackSolution(problem=self.problem, list_taken=list_taken)
