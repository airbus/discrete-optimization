#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

import matplotlib.pyplot as plt

from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.tsp.parser import get_data_available, parse_file
from discrete_optimization.tsp.plot import plot_tsp_solution
from discrete_optimization.tsp.solvers.optal import OptalTspSolver

logging.basicConfig(level=logging.INFO)


def run_optal_solver():
    files = get_data_available()
    files = [f for f in files if "tsp_100_1" in f]
    model = parse_file(files[0], start_index=0, end_index=0)
    params_objective_function = get_default_objective_setup(problem=model)
    solver = OptalTspSolver(model, params_objective_function=params_objective_function)
    solver.init_model(scaling=1)
    p = ParametersCp.default_cpsat()
    res = solver.solve(
        time_limit=10,
        parameters_cp=p,
        **{
            "worker0-1.searchType": "fdslb",
            "worker0-1.noOverlapPropagationLevel": 4,
            "worker0-1.cumulPropagationLevel": 3,
        },
    )
    sol, fitness = res.get_best_solution_fit()
    assert model.satisfy(sol)
    print(model.satisfy(sol), model.evaluate(sol))


if __name__ == "__main__":
    run_optal_solver()
