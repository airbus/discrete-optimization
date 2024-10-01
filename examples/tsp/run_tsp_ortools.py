#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

import matplotlib.pyplot as plt
import pytest

from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.generic_tools.lns_cp import LNS_OrtoolsCPSat
from discrete_optimization.generic_tools.lns_tools import ConstraintHandlerMix
from discrete_optimization.pickup_vrp.gpdp import ProxyClass
from discrete_optimization.pickup_vrp.solver.ortools_solver import ORToolsGPDP
from discrete_optimization.tsp.plots.plot_tsp import plot_tsp_solution
from discrete_optimization.tsp.solver.solver_ortools import TSP_ORtools
from discrete_optimization.tsp.solver.tsp_cpsat_lns import (
    ConstraintHandlerSubpathTSP,
    ConstraintHandlerTSP,
)
from discrete_optimization.tsp.solver.tsp_cpsat_solver import CpSatTspSolver
from discrete_optimization.tsp.tsp_parser import get_data_available, parse_file

logging.basicConfig(level=logging.INFO)


def run_ortools():
    files = get_data_available()
    files = [f for f in files if "tsp_724_1" in f]
    model = parse_file(files[0], start_index=0, end_index=0)
    gpdp = ProxyClass.from_tsp_model_gpdp(tsp_model=model)
    solution = model.get_dummy_solution()
    params_objective_function = get_default_objective_setup(problem=gpdp)
    solver = ORToolsGPDP(gpdp, params_objective_function=params_objective_function)
    solver.init_model(time_limit=100, include_time_dimension=False)
    callbacks = [
        ObjectiveLogger(
            step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
        )
    ]
    sol, fitness = solver.solve(callbacks=callbacks).get_best_solution_fit()
    assert gpdp.satisfy(sol)


if __name__ == "__main__":
    run_ortools()
