#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest

from discrete_optimization.generic_tools.cp_tools import CpSolverName, ParametersCp
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solvers.cp_mzn import (
    CpMultimodeRcpspSolver,
    CpRcpspSolver,
)


@pytest.mark.parametrize(
    "optimisation_level",
    [
        0,
        1,
        2,
        # 3, # fails with unbounded variable with gecode
    ],
)
def test_cp_sm(optimisation_level):
    files_available = get_data_available()
    file = [f for f in files_available if "j301_1.sm" in f][0]
    rcpsp_problem = parse_file(file)
    solver = CpMultimodeRcpspSolver(rcpsp_problem, cp_solver_name=CpSolverName.CHUFFED)
    solver.init_model(
        output_type=True,
        model_type="multi-resource-feasibility",
        max_time=rcpsp_problem.horizon,
    )
    params_cp = ParametersCp.default()
    params_cp.free_search = True
    params_cp.optimisation_level = optimisation_level
    solver.solve(parameters_cp=params_cp, time_limit=10)


@pytest.mark.parametrize(
    "optimisation_level",
    [
        0,
        1,
        2,
        # 3, # fails with unbounded variable with gecode
    ],
)
def test_cp_single_mode_model(optimisation_level):
    import logging

    logging.basicConfig(level=logging.DEBUG)
    files_available = get_data_available()
    file = [f for f in files_available if "j301_1.sm" in f][0]
    rcpsp_problem = parse_file(file)
    solver = CpRcpspSolver(rcpsp_problem, cp_solver_name=CpSolverName.CHUFFED)
    solver.init_model(
        output_type=True,
        model_type="single_resource",
        max_time=rcpsp_problem.horizon,
    )
    params_cp = ParametersCp.default()
    params_cp.free_search = True
    params_cp.optimisation_level = optimisation_level
    solver.solve(parameters_cp=params_cp, time_limit=10)
