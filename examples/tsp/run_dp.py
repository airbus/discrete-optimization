#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

import matplotlib.pyplot as plt

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.tsp.parser import get_data_available, parse_file
from discrete_optimization.tsp.plot import plot_tsp_solution
from discrete_optimization.tsp.solvers.dp import DpTspSolver, dp

logging.basicConfig(level=logging.INFO)


def run_dp_solver():
    files = get_data_available()
    files = [f for f in files if "tsp_574_1" in f]
    model = parse_file(files[0], start_index=0, end_index=0)
    params_objective_function = get_default_objective_setup(problem=model)
    solver = DpTspSolver(model, params_objective_function=params_objective_function)
    solver.init_model()
    res = solver.solve(
        solver=dp.LNBS, retrieve_intermediate_solutions=True, time_limit=50
    )
    sol, fitness = res.get_best_solution_fit()
    assert model.satisfy(sol)
    fig, ax = plt.subplots(1)
    for sol, fit in res.list_solution_fits:
        ax.clear()
        plot_tsp_solution(tsp_model=model, solution=sol, ax=ax)
        ax.set_title(f"Length ={fit}")
        plt.pause(1.0)
    plt.show()


def run_dp_solver_ws():
    files = get_data_available()
    files = [f for f in files if "tsp_574_1" in f]
    model = parse_file(files[0], start_index=0, end_index=0)
    params_objective_function = get_default_objective_setup(problem=model)
    from discrete_optimization.tsp.solvers.cpsat import CpSatTspSolver
    from discrete_optimization.tsp.solvers.gpdp import GpdpBasedTspSolver

    solver_ws = GpdpBasedTspSolver(model)
    sol = solver_ws.solve(
        time_limit=25, callbacks=[ObjectiveLogger(step_verbosity_level=logging.INFO)]
    )[-1][0]
    solver = DpTspSolver(model, params_objective_function=params_objective_function)
    solver.init_model()
    solver.set_warm_start(sol)
    res = solver.solve(
        solver=dp.LNBS,
        callbacks=[NbIterationStopper(nb_iteration_max=500)],
        retrieve_intermediate_solutions=True,
        threads=5,
        time_limit=40,
    )
    sol_, fitness = res.get_best_solution_fit()
    assert model.satisfy(sol_)
    assert res[0][0].permutation == sol.permutation
    fig, ax = plt.subplots(1)
    for sol, fit in res.list_solution_fits:
        ax.clear()
        plot_tsp_solution(tsp_model=model, solution=sol, ax=ax)
        ax.set_title(f"Length ={fit}")
        plt.pause(0.2)
    plt.show()


if __name__ == "__main__":
    # run_dp_solver()
    run_dp_solver_ws()
