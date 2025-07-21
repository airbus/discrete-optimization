#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from enum import Enum

from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.workforce.scheduling.problem import AllocSchedulingProblem


class ObjectivesEnum(Enum):
    NB_TEAMS = 1
    DISPERSION = 2
    MAKESPAN = 3
    MIN_WORKLOAD = 4
    NB_DONE_AC = 5  # number of done activities
    DELTA_TO_EXISTING_SOLUTION = 6


class SolverAllocScheduling(SolverDO):
    problem: AllocSchedulingProblem
