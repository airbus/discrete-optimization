#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random
import sys

import numpy as np
import pytest

from discrete_optimization.rcpsp.solver.cp_solvers import (
    CP_MRCPSP_MZN,
    CPSolverName,
    ParametersCP,
)
from discrete_optimization.rcpsp.solver.rcpsp_cp_lns_solver import (
    LNS_CP_RCPSP_SOLVER,
    OptionNeighbor,
)
from discrete_optimization.rcpsp_multiskill.multiskill_to_rcpsp import MultiSkillToRCPSP
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill_parser import (
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
    algorithm = MultiSkillToRCPSP(model_msrcpsp)
    rcpsp_model = algorithm.construct_rcpsp_by_worker_type(
        limit_number_of_mode_per_task=False,
        check_resource_compliance=True,
        max_number_of_mode=100,
    )
    assert rcpsp_model.n_jobs == 202


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Much too long on windows")
def test_solve_rcpsp_imopse1(random_seed):
    files = [f for f in get_data_available() if "200_40_133_15.def" in f]
    model_msrcpsp, new_name_to_original_task_id = parse_file(files[0], max_horizon=2000)
    algorithm = MultiSkillToRCPSP(model_msrcpsp)
    rcpsp_model = algorithm.construct_rcpsp_by_worker_type(
        limit_number_of_mode_per_task=True,
        check_resource_compliance=True,
        max_number_of_mode=5,
        one_worker_type_per_task=True,
    )
    params_cp = ParametersCP.default()
    params_cp.time_limit = 50
    params_cp.time_limit_iter0 = 100
    params_cp.free_search = False
    lns_solver = LNS_CP_RCPSP_SOLVER(
        rcpsp_model=rcpsp_model, option_neighbor=OptionNeighbor.MIX_FAST
    )
    result_storage = lns_solver.solve(
        parameters_cp=params_cp,
        nb_iteration_lns=5,
        max_time_seconds=100,
        nb_iteration_no_improvement=100,
        skip_first_iteration=False,
    )
    best_solution = result_storage.get_best_solution()
    assert rcpsp_model.satisfy(best_solution)


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Much too long on windows")
def test_solve_rcpsp_imopse2(random_seed):
    files = [f for f in get_data_available() if "200_40_133_15.def" in f]
    model_msrcpsp, new_name_to_original_task_id = parse_file(files[0], max_horizon=2000)
    algorithm = MultiSkillToRCPSP(model_msrcpsp)
    rcpsp_model = algorithm.construct_rcpsp_by_worker_type(
        limit_number_of_mode_per_task=True,
        check_resource_compliance=True,
        max_number_of_mode=5,
        one_worker_type_per_task=True,
    )
    params_cp = ParametersCP.default()
    params_cp.time_limit = 200
    params_cp.time_limit_iter0 = 300

    solver = CP_MRCPSP_MZN(rcpsp_model=rcpsp_model, cp_solver_name=CPSolverName.CHUFFED)
    solver.init_model(output_type=True)
    params_cp = ParametersCP.default()
    params_cp.time_limit = 100
    params_cp.free_search = True
    result_storage = solver.solve(parameters_cp=params_cp)
    best_solution = result_storage.get_best_solution()
    assert rcpsp_model.satisfy(best_solution)
