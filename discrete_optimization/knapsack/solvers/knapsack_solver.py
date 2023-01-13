#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any

from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.knapsack.knapsack_model import KnapsackModel


class SolverKnapsack(SolverDO):
    def __init__(self, knapsack_model: KnapsackModel, **kwargs: Any):
        self.knapsack_model = knapsack_model

    def init_model(self, **kwargs: Any) -> None:
        pass
