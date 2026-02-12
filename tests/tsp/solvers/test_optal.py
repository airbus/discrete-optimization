#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import pytest

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.tsp.parser import get_data_available, parse_file
from discrete_optimization.tsp.solvers.optal import ModelingTspEnum, OptalTspSolver


@pytest.mark.parametrize("modeling", (ModelingTspEnum.V0, ModelingTspEnum.V1))
def test_optal(modeling):
    files = get_data_available()
    files = [f for f in files if "tsp_100_1" in f]
    problem = parse_file(files[0], start_index=0, end_index=0)
    solver = OptalTspSolver(problem=problem)
    solver.init_model(scaling=1, modeling=modeling)
    res = solver.solve(
        parameters_cp=ParametersCp.default_cpsat(),
        time_limit=10,
    )
    assert solver.status_solver in (StatusSolver.OPTIMAL, StatusSolver.SATISFIED)
