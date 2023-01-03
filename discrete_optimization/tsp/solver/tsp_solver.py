#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any

from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.tsp.tsp_model import TSPModel


class SolverTSP(SolverDO):
    def __init__(self, tsp_model: TSPModel, **kwargs: Any):
        self.tsp_model = tsp_model

    def init_model(self, **kwargs: Any) -> None:
        pass
