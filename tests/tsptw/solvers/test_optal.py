#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest

from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.tsptw.solvers.optal import (
    OptalTspTwSolver,
    optalcp_available,
)


@pytest.mark.skipif(
    not optalcp_available, reason="You need optalcp to test this solver."
)
def test_optal(problem):
    solver = OptalTspTwSolver(problem=problem)
    solver.init_model()
    res = solver.solve(
        time_limit=2,
        do_not_retrieve_solutions=True,  # optalcp-preview mode
    )
    assert solver.status_solver in (StatusSolver.OPTIMAL, StatusSolver.SATISFIED)
    assert solver.current_obj >= solver.current_bound
