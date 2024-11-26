#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

import didppy
import matplotlib.pyplot as plt

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
    TimerStopper,
)
from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.generic_tools.lns_tools import (
    BaseLns,
    ConstraintHandlerMix,
    TrivialInitialSolution,
    from_solutions_to_result_storage,
)
from discrete_optimization.generic_tools.toulbar_tools import to_lns_toulbar
from discrete_optimization.tsp.parser import get_data_available, parse_file
from discrete_optimization.tsp.plot import plot_tsp_solution
from discrete_optimization.tsp.solvers.dp import DpTspSolver
from discrete_optimization.tsp.solvers.toulbar import (
    ToulbarTspSolver,
    TspConstraintHandlerToulbar,
)

logging.basicConfig(level=logging.INFO)


def run_toulbar_solver():
    files = get_data_available()
    files = [f for f in files if "tsp_99_1" in f]
    model = parse_file(files[0], start_index=0, end_index=0)
    params_objective_function = get_default_objective_setup(problem=model)
    solver = ToulbarTspSolver(
        model, params_objective_function=params_objective_function
    )
    solver.init_model(vns=-4)
    solver.set_warm_start(model.get_dummy_solution())
    print("starting to solve")
    res = solver.solve(time_limit=45)
    sol, fitness = res.get_best_solution_fit()
    assert model.satisfy(sol)
    print(sol, fitness)
    fig, ax = plt.subplots(1)
    for sol, fit in res.list_solution_fits:
        ax.clear()
        plot_tsp_solution(tsp_model=model, solution=sol, ax=ax)
        ax.set_title(f"Length ={fit}")
        plt.pause(1.0)
    plt.show()


def run_toulbar_solver_ws():
    files = get_data_available()
    files = [f for f in files if "tsp_99_1" in f]
    model = parse_file(files[0], start_index=0, end_index=0)
    params_objective_function = get_default_objective_setup(problem=model)
    solver_ws = DpTspSolver(model)
    solver_ws.init_model()
    sol = solver_ws.solve(time_limit=5).get_best_solution_fit()[0]
    solver = ToulbarTspSolver(
        model, params_objective_function=params_objective_function
    )
    solver.init_model(vns=-3, encoding_all_diff="salldiffkp")
    solver.set_warm_start(sol)
    print("starting to solve")
    res = solver.solve(time_limit=45)
    sol, fitness = res.get_best_solution_fit()
    assert model.satisfy(sol)
    print(sol, fitness)
    fig, ax = plt.subplots(1)
    for sol, fit in res.list_solution_fits:
        ax.clear()
        plot_tsp_solution(tsp_model=model, solution=sol, ax=ax)
        ax.set_title(f"Length ={fit}")
        plt.pause(1.0)
    plt.show()


def run_toulbar_solver_lns():
    files = get_data_available()
    files = [f for f in files if "tsp_299_1" in f]
    model = parse_file(files[0], start_index=0, end_index=0)
    params_objective_function = get_default_objective_setup(problem=model)
    solver_ws = DpTspSolver(model)
    solver_ws.init_model()
    sol = solver_ws.solve(time_limit=3, solver=didppy.CABS).get_best_solution_fit()[0]

    solver = to_lns_toulbar(ToulbarTspSolver)(
        model, params_objective_function=params_objective_function
    )
    solver.init_model(vns=None, encoding_all_diff="salldiffkp")
    solver_lns = BaseLns(
        problem=model,
        initial_solution_provider=TrivialInitialSolution(
            from_solutions_to_result_storage([sol], problem=model)
        ),
        constraint_handler=TspConstraintHandlerToulbar(fraction_nodes=0.85),
        subsolver=solver,
    )
    print("starting to solve")
    res = solver_lns.solve(
        nb_iteration_lns=1000,
        skip_initial_solution_provider=False,
        time_limit_subsolver=5,
        callbacks=[TimerStopper(total_seconds=100)],
    )
    sol, fitness = res.get_best_solution_fit()
    assert model.satisfy(sol)
    print(sol, fitness)
    fig, ax = plt.subplots(1)
    for sol, fit in res.list_solution_fits[-10:]:
        ax.clear()
        plot_tsp_solution(tsp_model=model, solution=sol, ax=ax)
        ax.set_title(f"Length ={fit}")
        plt.pause(1.0)
    plt.show()


if __name__ == "__main__":
    run_toulbar_solver_lns()
