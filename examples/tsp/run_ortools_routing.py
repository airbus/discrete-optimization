#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.gpdp.problem import ProxyClass
from discrete_optimization.gpdp.solvers.ortools_routing import OrtoolsGpdpSolver
from discrete_optimization.tsp.parser import get_data_available, parse_file

logging.basicConfig(level=logging.INFO)


def run_ortools():
    files = get_data_available()
    files = [f for f in files if "tsp_724_1" in f]
    model = parse_file(files[0], start_index=0, end_index=0)
    gpdp = ProxyClass.from_tsp_to_gpdp(tsp_model=model)
    solution = model.get_dummy_solution()
    params_objective_function = get_default_objective_setup(problem=gpdp)
    solver = OrtoolsGpdpSolver(
        gpdp, params_objective_function=params_objective_function
    )
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
