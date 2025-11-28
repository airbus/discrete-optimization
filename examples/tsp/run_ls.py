#  Copyright (c) 2024-2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from matplotlib import pyplot as plt

from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.generic_tools.ls.local_search import (
    ModeMutation,
    RestartHandlerLimit,
)
from discrete_optimization.generic_tools.ls.simulated_annealing import (
    SimulatedAnnealing,
    TemperatureSchedulingFactor,
)
from discrete_optimization.generic_tools.mutations.mutation_portfolio import (
    create_mutations_portfolio_from_problem,
)
from discrete_optimization.tsp.mutation import SwapTspMutation, TwoOptTspMutation
from discrete_optimization.tsp.parser import get_data_available, parse_file
from discrete_optimization.tsp.plot import plot_tsp_solution

logging.basicConfig(level=logging.INFO)


def run_sa():
    files = get_data_available()
    files = [f for f in files if "tsp_100_1" in f]
    model = parse_file(files[0])
    params_objective_function = get_default_objective_setup(problem=model)
    solution = model.get_dummy_solution()
    res = RestartHandlerLimit(3000)
    portfolio_mutation = create_mutations_portfolio_from_problem(
        problem=model,
        selected_mutations={SwapTspMutation, TwoOptTspMutation},
        selected_attributes={"permutation"},
    )
    sa = SimulatedAnnealing(
        problem=model,
        mutator=portfolio_mutation,
        restart_handler=res,
        temperature_handler=TemperatureSchedulingFactor(
            temperature=100, restart_handler=res, coefficient=0.99999
        ),
        mode_mutation=ModeMutation.MUTATE_AND_EVALUATE,
        params_objective_function=params_objective_function,
    )
    res = sa.solve(
        callbacks=[
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
        initial_variable=solution,
        nb_iteration_max=100000,
    )
    assert model.satisfy(res.get_best_solution())
    fig, ax = plt.subplots(1)
    list_solution_fit = sorted(res.list_solution_fits, key=lambda x: x[1], reverse=True)
    for sol, fit in list_solution_fit:
        ax.clear()
        plot_tsp_solution(tsp_model=model, solution=sol, ax=ax)
        ax.set_title(f"Length ={fit}")
        plt.pause(0.05)
    plt.show()


if __name__ == "__main__":
    run_sa()
