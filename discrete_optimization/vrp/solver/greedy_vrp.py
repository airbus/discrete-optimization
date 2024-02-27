#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any

from discrete_optimization.generic_tools.do_solver import ResultStorage
from discrete_optimization.vrp.solver.vrp_solver import SolverVrp
from discrete_optimization.vrp.vrp_model import VrpProblem, trivial_solution


class GreedyVRPSolver(SolverVrp):
    def solve(self, **kwargs: Any) -> ResultStorage:
        sol, _ = trivial_solution(self.problem)
        fit = self.aggreg_from_sol(sol)
        return ResultStorage(
            list_solution_fits=[(sol, fit)],
            mode_optim=self.params_objective_function.sense_function,
        )
