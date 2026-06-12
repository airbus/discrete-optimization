#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest

from discrete_optimization.flex_scheduling.generator import FlexProblemGenerator
from discrete_optimization.flex_scheduling.solvers.optal import (
    OptalFlexProblemSolver,
    optalcp_available,
)
from discrete_optimization.generic_tools.do_solver import StatusSolver


@pytest.mark.skipif(
    not optalcp_available, reason="You need optalcp to test this solver."
)
def test_optal_simple():
    """Test OptalCP solver on a simple generated problem."""
    generator = FlexProblemGenerator(
        nb_msn=2,  # Small problem
        seed=42,
        tardiness_weight=10,
        earliness_weight=1,
        nb_tools=2,
        nb_stations=8,
    )
    problem = generator.generate()

    solver = OptalFlexProblemSolver(problem=problem)
    solver.init_model()
    res = solver.solve(
        time_limit=5,
        do_not_retrieve_solutions=True,  # optalcp-preview mode
    )
    assert solver.status_solver in (StatusSolver.OPTIMAL, StatusSolver.SATISFIED)
    assert solver.current_obj >= solver.current_bound
