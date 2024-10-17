#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

import matplotlib.pyplot as plt

from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.generic_tools.lns_cp import LnsOrtoolsCpSat
from discrete_optimization.generic_tools.lns_tools import ConstraintHandlerMix
from discrete_optimization.tsp.parser import get_data_available, parse_file
from discrete_optimization.tsp.plot import plot_tsp_solution
from discrete_optimization.tsp.solvers.cpsat import CpSatTspSolver
from discrete_optimization.tsp.solvers.lns_cpsat import (
    SubpathTspConstraintHandler,
    TspConstraintHandler,
)

logging.basicConfig(level=logging.INFO)


def run_cpsat_lns_solver():
    files = get_data_available()
    files = [f for f in files if "tsp_299_1" in f]
    model = parse_file(files[0], start_index=0, end_index=10)
    params_objective_function = get_default_objective_setup(problem=model)
    solver = CpSatTspSolver(model, params_objective_function=params_objective_function)
    solver.init_model()
    p = ParametersCp.default_cpsat()
    lns_solver = LnsOrtoolsCpSat(
        problem=model,
        subsolver=solver,
        constraint_handler=ConstraintHandlerMix(
            problem=model,
            list_constraints_handler=[
                SubpathTspConstraintHandler(problem=model, fraction_segment_to_fix=0.7),
                TspConstraintHandler(problem=model, fraction_segment_to_fix=0.7),
            ],
            list_proba=[0.5, 0.5],
        ),
    )
    res = lns_solver.solve(
        skip_initial_solution_provider=True,
        nb_iteration_lns=1000,
        callbacks=[
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
        parameters_cp=p,
        time_limit_subsolver_iter0=10,
        time_limit_subsolver=10,
    )
    sol, fitness = res.get_best_solution_fit()
    assert model.satisfy(sol)
    fig, ax = plt.subplots(1)
    list_solution_fit = sorted(res.list_solution_fits, key=lambda x: x[1], reverse=True)
    for sol, fit in list_solution_fit:
        ax.clear()
        plot_tsp_solution(tsp_model=model, solution=sol, ax=ax)
        ax.set_title(f"Length ={fit}")
        plt.pause(0.05)
    plt.show()


if __name__ == "__main__":
    run_cpsat_lns_solver()
