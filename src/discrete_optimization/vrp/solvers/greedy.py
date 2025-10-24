#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any

from discrete_optimization.generic_tools.do_solver import ResultStorage
from discrete_optimization.vrp.problem import trivial_solution
from discrete_optimization.vrp.solvers import VrpSolver


class GreedyVrpSolver(VrpSolver):
    def solve(self, **kwargs: Any) -> ResultStorage:
        sol, _ = trivial_solution(self.problem)
        fit = self.aggreg_from_sol(sol)
        return self.create_result_storage(
            [(sol, fit)],
        )
