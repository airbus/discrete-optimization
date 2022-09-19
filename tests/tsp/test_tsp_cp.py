#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import sys

import pytest

from discrete_optimization.generic_tools.cp_tools import CPSolverName, ParametersCP
from discrete_optimization.tsp.solver.tsp_cp_solver import TSP_CP_Solver, TSP_CPModel
from discrete_optimization.tsp.tsp_parser import get_data_available, parse_file


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Much too long on windows")
def test_int_cp():
    files = get_data_available()
    files = [f for f in files if "tsp_100_3" in f]
    model = parse_file(files[0], start_index=0, end_index=10)
    model_type = TSP_CPModel.INT_VERSION
    cp_solver = TSP_CP_Solver(
        model, model_type=model_type, cp_solver_name=CPSolverName.CHUFFED
    )
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 20
    cp_solver.init_model()
    var, fit = cp_solver.solve(parameters_cp=parameters_cp).get_best_solution_fit()
    assert model.satisfy(var)


def test_float_cp():
    files = get_data_available()
    files = [f for f in files if "tsp_100_3" in f]
    model = parse_file(files[0], start_index=0, end_index=10)
    model_type = TSP_CPModel.FLOAT_VERSION
    cp_solver = TSP_CP_Solver(
        model, model_type=model_type, cp_solver_name=CPSolverName.GECODE
    )  # CHUFFED WONT WORK FOR FLOAT
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 20
    cp_solver.init_model()
    var, fit = cp_solver.solve(parameters_cp=parameters_cp).get_best_solution_fit()
    assert model.satisfy(var)


if __name__ == "__main__":
    test_int_cp()
