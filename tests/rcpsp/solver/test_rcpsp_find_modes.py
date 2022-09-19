#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tools.cp_tools import CPSolverName, ParametersCP
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solver.cp_solvers import CP_MRCPSP_MZN_MODES


def test_find_modes():
    files_available = get_data_available()
    file = [f for f in files_available if "j1010_1.mm" in f][0]
    rcpsp_problem = parse_file(file)
    solver = CP_MRCPSP_MZN_MODES(rcpsp_problem, cp_solver_name=CPSolverName.CHUFFED)
    params_cp = ParametersCP.default()
    params_cp.nr_solutions = float("inf")
    params_cp.all_solutions = True
    result_storage = solver.solve(parameters_cp=params_cp)
    assert len(result_storage) == 12744


if __name__ == "__main__":
    test_find_modes()
