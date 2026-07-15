#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Base MILP solver for generic lot sizing problems."""

from discrete_optimization.generic_tools.toulbar_tools import ToulbarSolver
from discrete_optimization.lotsizing.generic_solver.lotsizing_solver import (
    Item,
    LotSizingGenericSolver,
)


class LotSizingToulbarSolver(LotSizingGenericSolver[Item], ToulbarSolver):
    """Base Toulbar solver combining generic lot sizing solver interface with MILP capabilities.

    This class serves as the base for all Toulbar-based lot sizing solvers.
    It inherits from:
    - LotSizingGenericSolver: provides lot sizing problem interface
    - ToulbarSolver: provides abstract Toulbar solver interface
    """

    pass
