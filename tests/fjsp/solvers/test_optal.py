#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest

import discrete_optimization.fjsp.parser as fjsp_parser
from discrete_optimization.fjsp.solvers.optal import OptalFJspSolver, optalcp_available
from discrete_optimization.generic_tools.do_solver import StatusSolver


@pytest.mark.skipif(
    not optalcp_available, reason="You need optalcp to test this solver."
)
def test_optal():
    files = fjsp_parser.get_data_available()
    file = [f for f in files if "Behnke1.fjs" in f][0]
    problem = fjsp_parser.parse_file(file)
    solver = OptalFJspSolver(problem=problem)
    solver.init_model()
    res = solver.solve(
        time_limit=2,
        do_not_retrieve_solutions=True,  # optalcp-preview mode
    )
    assert solver.status_solver in (StatusSolver.OPTIMAL, StatusSolver.SATISFIED)
    assert solver.current_obj >= solver.current_bound
