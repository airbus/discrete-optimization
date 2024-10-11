#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import sys

import pytest

from discrete_optimization.generic_tools.cp_tools import CpSolverName, ParametersCp
from discrete_optimization.tsp.parser import get_data_available, parse_file
from discrete_optimization.tsp.solvers.cp_mzn import CPTspModel, CpTspSolver


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Much too long on windows")
def test_int_cp():
    files = get_data_available()
    files = [f for f in files if "tsp_100_3" in f]
    model = parse_file(files[0], start_index=0, end_index=10)
    model_type = CPTspModel.INT_VERSION
    cp_solver = CpTspSolver(
        model, model_type=model_type, cp_solver_name=CpSolverName.CHUFFED
    )
    cp_solver.init_model()
    var, fit = cp_solver.solve(time_limit=20).get_best_solution_fit()
    assert model.satisfy(var)


def test_float_cp():
    files = get_data_available()
    files = [f for f in files if "tsp_100_3" in f]
    model = parse_file(files[0], start_index=0, end_index=10)
    model_type = CPTspModel.FLOAT_VERSION
    cp_solver = CpTspSolver(
        model, model_type=model_type, cp_solver_name=CpSolverName.GECODE
    )  # CHUFFED WONT WORK FOR FLOAT
    cp_solver.init_model()
    var, fit = cp_solver.solve(time_limit=20).get_best_solution_fit()
    assert model.satisfy(var)


if __name__ == "__main__":
    test_int_cp()
