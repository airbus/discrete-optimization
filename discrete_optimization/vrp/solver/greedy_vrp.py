#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any, Optional

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.do_solver import ResultStorage
from discrete_optimization.vrp.solver.vrp_solver import SolverVrp
from discrete_optimization.vrp.vrp_model import VrpProblem, trivial_solution


class GreedyVRPSolver(SolverVrp):
    def __init__(
        self,
        vrp_model: VrpProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
    ):
        SolverVrp.__init__(self, vrp_model=vrp_model)
        (
            self.aggreg_sol,
            self.aggreg_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.vrp_model, params_objective_function=params_objective_function
        )

    def solve(self, **kwargs: Any) -> ResultStorage:
        sol, _ = trivial_solution(self.vrp_model)
        fit = self.aggreg_sol(sol)
        return ResultStorage(
            list_solution_fits=[(sol, fit)],
            mode_optim=self.params_objective_function.sense_function,
        )
