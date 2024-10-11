#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Union

from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp.problem_preemptive import PreemptiveRcpspProblem
from discrete_optimization.rcpsp.problem_specialized_constraints import (
    SpecialConstraintsPreemptiveRcpspProblem,
)

GENERIC_CLASS = Union[
    RcpspProblem,
    PreemptiveRcpspProblem,
    SpecialConstraintsPreemptiveRcpspProblem,
]
