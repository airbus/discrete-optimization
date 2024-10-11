#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random
import sys

import numpy as np
import pytest

from discrete_optimization.generic_tools.cp_tools import CpSolverName, ParametersCp
from discrete_optimization.rcpsp.solvers.cp_mzn import CpMultimodeRcpspSolver
from discrete_optimization.rcpsp_multiskill.multiskill_to_rcpsp import MultiSkillToRcpsp
from discrete_optimization.rcpsp_multiskill.parser_imopse import (
    get_data_available,
    parse_file,
)


@pytest.fixture
def random_seed():
    random.seed(42)
    np.random.seed(42)


def test_ms_to_rcpsp_imopse(random_seed):
    files = [f for f in get_data_available() if "200_40_133_15.def" in f]
    model_msrcpsp, new_name_to_original_task_id = parse_file(files[0], max_horizon=2000)
    algorithm = MultiSkillToRcpsp(model_msrcpsp)
    rcpsp_problem = algorithm.construct_rcpsp_by_worker_type(
        limit_number_of_mode_per_task=False,
        check_resource_compliance=True,
        max_number_of_mode=100,
    )
    assert rcpsp_problem.n_jobs == 202


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Much too long on windows")
def test_solve_rcpsp_imopse2(random_seed):
    files = [f for f in get_data_available() if "200_40_133_15.def" in f]
    model_msrcpsp, new_name_to_original_task_id = parse_file(files[0], max_horizon=2000)
    algorithm = MultiSkillToRcpsp(model_msrcpsp)
    rcpsp_problem = algorithm.construct_rcpsp_by_worker_type(
        limit_number_of_mode_per_task=True,
        check_resource_compliance=True,
        max_number_of_mode=5,
        one_worker_type_per_task=True,
    )
    solver = CpMultimodeRcpspSolver(
        problem=rcpsp_problem, cp_solver_name=CpSolverName.CHUFFED
    )
    solver.init_model(output_type=True)
    params_cp = ParametersCp.default()
    params_cp.free_search = True
    result_storage = solver.solve(parameters_cp=params_cp, time_limit=100)
    best_solution = result_storage.get_best_solution()
    assert rcpsp_problem.satisfy(best_solution)
