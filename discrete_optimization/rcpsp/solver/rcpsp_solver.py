#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any, Union

from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel
from discrete_optimization.rcpsp.rcpsp_model_preemptive import RCPSPModelPreemptive


class SolverRCPSP(SolverDO):
    def __init__(
        self, rcpsp_model: Union[RCPSPModel, RCPSPModelPreemptive], **kwargs: Any
    ):
        self.rcpsp_model = rcpsp_model

    def init_model(self, **kwargs: Any) -> None:
        pass
