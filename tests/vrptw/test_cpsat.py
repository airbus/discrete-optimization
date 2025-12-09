#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.vrptw.parser import get_data_available, parse_vrptw_file
from discrete_optimization.vrptw.solvers.cpsat import CpSatVRPTWSolver


def test_cpsat():
    file = [f for f in get_data_available() if "C1_2_1.TXT" in f][0]
    problem = parse_vrptw_file(file)
    solver = CpSatVRPTWSolver(problem=problem)
    solver.init_model(scaling=10, cost_per_vehicle=1000)
    res = solver.solve(
        callbacks=[NbIterationStopper(1)],
        parameters_cp=ParametersCp.default_cpsat(),
        time_limit=100,
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
    )
    sol = res[-1][0]
    # assert problem.satisfy(sol)
