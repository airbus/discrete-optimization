#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.gpdp.problem import GpdpProblem


class GpdpSolver(SolverDO):
    problem: GpdpProblem
