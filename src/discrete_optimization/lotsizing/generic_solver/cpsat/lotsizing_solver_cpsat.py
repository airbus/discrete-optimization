#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver
from discrete_optimization.lotsizing.generic_solver.lotsizing_solver import (
    Item,
    LotSizingGenericSolver,
)


class LotSizingCpSatSolver(LotSizingGenericSolver[Item], OrtoolsCpSatSolver): ...
