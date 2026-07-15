#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Base MILP solver for generic lot sizing problems."""

from discrete_optimization.generic_tools.lp_tools import (
    MilpSolver,
)
from discrete_optimization.lotsizing.generic_solver.lotsizing_solver import (
    Item,
    LotSizingGenericSolver,
)


class LotSizingMilpSolver(LotSizingGenericSolver[Item], MilpSolver):
    """Base MILP solver combining generic lot sizing solver interface with MILP capabilities.

    This class serves as the base for all MILP-based lot sizing solvers.
    It inherits from:
    - LotSizingGenericSolver: provides lot sizing problem interface
    - MilpSolver: provides abstract MILP solver interface

    Concrete implementations should inherit from this class and either
    OrtoolsMathOptMilpSolver or GurobiMilpSolver.
    """

    pass
