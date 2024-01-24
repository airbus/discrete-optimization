#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any, Optional

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.vrp.vrp_model import VrpProblem


class SolverVrp(SolverDO):
    def __init__(
        self,
        vrp_model: VrpProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any
    ):
        self.vrp_model = vrp_model
        (
            self.aggreg_sol,
            self.aggreg_from_dict_values,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            self.vrp_model, params_objective_function=params_objective_function
        )

    def init_model(self, **kwargs: Any) -> None:
        pass
