#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Solvers for the Oven Scheduling Problem."""

from discrete_optimization.ovensched.solvers.chunking import (
    ChunkingOvenSchedulingSolver,
    ChunkingParams,
    ChunkingStrategy,
)
from discrete_optimization.ovensched.solvers.cpsat import OvenSchedulingCpSatSolver
from discrete_optimization.ovensched.solvers.greedy import GreedyOvenSchedulingSolver

__all__ = [
    "OvenSchedulingCpSatSolver",
    "GreedyOvenSchedulingSolver",
    "ChunkingOvenSchedulingSolver",
    "ChunkingParams",
    "ChunkingStrategy",
]

if optalcp_available:
    __all__.append("OvenSchedulingOptalSolver")
