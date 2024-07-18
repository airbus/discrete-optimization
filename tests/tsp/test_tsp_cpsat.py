#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

import pytest

from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.tsp.solver.tsp_cpsat_solver import CpSatTspSolver
from discrete_optimization.tsp.tsp_parser import get_data_available, parse_file


def test_cpsat_solver():
    files = get_data_available()
    files = [f for f in files if "tsp_100_1" in f]
    model = parse_file(files[0], start_index=0, end_index=10)
    params_objective_function = get_default_objective_setup(problem=model)
    solver = CpSatTspSolver(model, params_objective_function=params_objective_function)
    solver.init_model()
    res = solver.solve(
        callbacks=[
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
        parameters_cp=ParametersCP.default_cpsat(),
    )
    sol, fitness = res.get_best_solution_fit()
    assert model.satisfy(sol)
