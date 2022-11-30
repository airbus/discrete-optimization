#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  In this example script we're parsing dzn files that are retrieved from https://github.com/youngkd/MSPSP-InstLib
#  And run CP solver with different mzn models.

import logging

from discrete_optimization.generic_tools.cp_tools import CPSolverName, ParametersCP
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill_mspsp_parser import (
    get_data_available_mspsp,
    parse_dzn_file,
)
from discrete_optimization.rcpsp_multiskill.solvers.cp_solver_mspsp_instlib import (
    CP_MSPSP_MZN,
)


def test_solve():
    logging.basicConfig(level=logging.DEBUG)
    datas_dict = get_data_available_mspsp()
    some_file = datas_dict["set-1"]["set-1a"][0]
    model = parse_dzn_file(file_path=some_file)
    cp_solver_1 = CP_MSPSP_MZN(model, cp_solver_name=CPSolverName.CHUFFED)
    cp_solver_1.init_model(
        model_type="mspsp",
        output_type=True,
        ignore_sec_objective=True,
        add_objective_makespan=True,
    )
    cp_solver_1.instance["maxt"] = 150
    cp_solver_1.instance["full_output"] = True
    cp_solver_1.instance.add_string("my_search=priority_smallest;\n")
    parameters_cp = ParametersCP.default()
    parameters_cp.TimeLimit = 35
    parameters_cp.TimeLimit_iter0 = 25
    parameters_cp.multiprocess = False
    parameters_cp.nb_process = 4
    results = cp_solver_1.solve(parameters_cp=parameters_cp)
    solution = results.get_best_solution()
    assert model.satisfy(solution)
    cp_solver_2 = CP_MSPSP_MZN(model, cp_solver_name=CPSolverName.CHUFFED)
    cp_solver_2.init_model(
        model_type="mspsp_compatible",
        output_type=True,
        ignore_sec_objective=True,
        add_objective_makespan=True,
    )
    cp_solver_2.instance["maxt"] = 150
    cp_solver_2.instance["full_output"] = True
    parameters_cp = ParametersCP.default()
    parameters_cp.TimeLimit = 35
    parameters_cp.TimeLimit_iter0 = 25
    parameters_cp.multiprocess = False
    parameters_cp.free_search = True
    results = cp_solver_2.solve(parameters_cp=parameters_cp)
    solution = results.get_best_solution()
    assert model.satisfy(solution)
