#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from typing import Any, List, Optional

import didppy as dp

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.dyn_prog_tools import DidSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.knapsack.knapsack_model import KnapsackSolution
from discrete_optimization.knapsack.solvers.knapsack_solver import SolverKnapsack

logger = logging.getLogger(__name__)


class DidKnapsackSolver(SolverKnapsack, DidSolver):
    hyperparameters = DidSolver.hyperparameters
    model: dp.Model

    def init_model(self, **kwargs):
        """Adapted from https://didppy.readthedocs.io/en/stable/quickstart.html"""
        n = self.problem.nb_items
        weights = [self.problem.list_items[i].weight for i in range(n)]
        profits = [self.problem.list_items[i].value for i in range(n)]
        capacity = self.problem.max_capacity
        model = dp.Model(maximize=True, float_cost=False)
        item = model.add_object_type(number=self.problem.nb_items)
        r = model.add_int_var(target=capacity)
        i = model.add_element_var(object_type=item, target=0)

        w = model.add_int_table(weights)
        p = model.add_int_table(profits)
        pack = dp.Transition(
            name="pack",
            cost=p[i] + dp.IntExpr.state_cost(),
            effects=[(r, r - w[i]), (i, i + 1)],
            preconditions=[i < n, r >= w[i]],
        )
        model.add_transition(pack)

        ignore = dp.Transition(
            name="ignore",
            cost=dp.IntExpr.state_cost(),
            effects=[(i, i + 1)],
            preconditions=[i < n],
        )
        model.add_transition(ignore)
        model.add_base_case([i == n])
        self.model = model

    def solve(
        self, callbacks: Optional[List[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        solver_cls = kwargs["solver"]
        solver = solver_cls(self.model, time_limit=10, quiet=False)
        solution = solver.search()
        taken = [0 for _ in range(self.problem.nb_items)]
        for i, t in enumerate(solution.transitions):
            if t.name == "pack":
                taken[i] = 1
                logger.info(f"pack {i}")
        logger.info(f"profit: {solution.cost}")
        sol = KnapsackSolution(problem=self.problem, list_taken=taken)
        fit = self.aggreg_from_sol(sol)
        return self.create_result_storage([(sol, fit)])
