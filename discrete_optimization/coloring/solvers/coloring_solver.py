#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.coloring.problem import ColoringProblem
from discrete_optimization.generic_tools.do_solver import SolverDO


class ColoringSolver(SolverDO):
    problem: ColoringProblem
