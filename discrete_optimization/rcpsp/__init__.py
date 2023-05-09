#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Union

from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel
from discrete_optimization.rcpsp.rcpsp_model_preemptive import RCPSPModelPreemptive
from discrete_optimization.rcpsp.specialized_rcpsp.rcpsp_specialized_constraints import (
    RCPSPModelSpecialConstraintsPreemptive,
)

GENERIC_CLASS = Union[
    RCPSPModel,
    RCPSPModelPreemptive,
    RCPSPModelSpecialConstraintsPreemptive,
]
