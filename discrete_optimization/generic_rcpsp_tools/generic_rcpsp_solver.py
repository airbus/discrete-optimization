#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any

from discrete_optimization.generic_rcpsp_tools.typing import ANY_RCPSP
from discrete_optimization.generic_tools.do_solver import SolverDO


class SolverGenericRCPSP(SolverDO):
    def __init__(self, rcpsp_model: ANY_RCPSP, **kwargs: Any):
        self.rcpsp_model = rcpsp_model

    def init_model(self, **kwargs: Any) -> None:
        pass
