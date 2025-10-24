#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import os

from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.workforce.allocation.problem import TeamAllocationProblem

folder_solver = os.path.dirname(os.path.abspath(__file__))


class TeamAllocationSolver(SolverDO):
    problem: TeamAllocationProblem
