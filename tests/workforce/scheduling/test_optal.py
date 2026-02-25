#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.workforce.scheduling.parser import (
    get_data_available,
    parse_json_to_problem,
)
from discrete_optimization.workforce.scheduling.solvers.optal import (
    OptalAllocSchedulingSolver,
    optalcp_available,
)


@pytest.mark.skipif(
    not optalcp_available, reason="You need optalcp to test this solver."
)
def test_optal():
    instance = [p for p in get_data_available() if "instance_64.json" in p][0]
    problem = parse_json_to_problem(instance)
    solver = OptalAllocSchedulingSolver(problem=problem)
    solver.init_model()
    res = solver.solve(
        parameters_cp=ParametersCp.default_cpsat(),
        time_limit=10,
        do_not_retrieve_solutions=True,  # optalcp-preview mode
    )
    assert solver.status_solver in (StatusSolver.OPTIMAL, StatusSolver.SATISFIED)
    assert solver.current_obj >= solver.current_bound
