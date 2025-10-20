#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.tsptw.solvers.cpsat import CpSatTSPTWSolver


@pytest.mark.parametrize("relaxed", [False, True])
def test_cpsat(problem, relaxed):
    parameters_cp = ParametersCp.default()
    solver = CpSatTSPTWSolver(problem=problem)
    solver.init_model(relaxed=relaxed)
    res = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)], parameters_cp=parameters_cp
    )
    sol = res.get_best_solution()
    if not relaxed:
        assert problem.satisfy(sol)
