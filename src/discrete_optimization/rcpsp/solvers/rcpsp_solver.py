#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Union

from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp.problem_preemptive import PreemptiveRcpspProblem


class RcpspSolver(SolverDO):
    problem: Union[RcpspProblem, PreemptiveRcpspProblem]
