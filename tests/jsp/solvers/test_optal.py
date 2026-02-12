#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest

from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.jsp.parser import get_data_available, parse_file
from discrete_optimization.jsp.solvers.optal import OptalJspSolver, optalcp_available


@pytest.mark.skipif(
    not optalcp_available, reason="You need optalcp to test this solver."
)
def test_optal():
    filename = "la02"
    filepath = [f for f in get_data_available() if f.endswith(filename)][0]
    problem = parse_file(filepath)
    solver = OptalJspSolver(problem=problem)
    solver.init_model()
    res = solver.solve(
        time_limit=5,
        do_not_retrieve_solutions=True,  # optalcp-preview mode
    )
    assert solver.status_solver in (StatusSolver.OPTIMAL, StatusSolver.SATISFIED)
    assert solver.current_obj >= solver.current_bound
