#  Copyright (c) 2022-2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import numpy as np

from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.generic_tools.ls.hill_climber import HillClimberPareto
from discrete_optimization.generic_tools.ls.local_search import RestartHandlerLimit
from discrete_optimization.generic_tools.ls.simulated_annealing import (
    ModeMutation,
    SimulatedAnnealing,
    TemperatureSchedulingFactor,
)
from discrete_optimization.generic_tools.mutations.mutation_catalog import (
    get_available_mutations,
)
from discrete_optimization.generic_tools.mutations.mutation_portfolio import (
    create_mutations_portfolio_from_problem,
)
from discrete_optimization.knapsack.mutation import BitFlipKnapsackMutation
from discrete_optimization.knapsack.parser import get_data_available, parse_file
from discrete_optimization.knapsack.problem import KnapsackProblem, MobjKnapsackModel


def test_sa_knapsack():
    model_file = [f for f in get_data_available() if "ks_60_0" in f][0]
    model: KnapsackProblem = parse_file(model_file, force_recompute_values=True)
    solution = model.get_dummy_solution()
    mixed_mutation = create_mutations_portfolio_from_problem(problem=model)
    res = RestartHandlerLimit(3000)
    sa = SimulatedAnnealing(
        problem=model,
        mutator=mixed_mutation,
        restart_handler=res,
        temperature_handler=TemperatureSchedulingFactor(1000, res, 0.99),
        mode_mutation=ModeMutation.MUTATE,
    )
    sa.solve(
        initial_variable=solution,
        nb_iteration_max=1000000,
        callbacks=[TimerStopper(total_seconds=20, check_nb_steps=1000)],
    )


def test_hc_knapsack_multiobj():
    model_file = [f for f in get_data_available() if "ks_60_0" in f][0]
    model: KnapsackProblem = parse_file(model_file, force_recompute_values=True)
    model: MobjKnapsackModel = MobjKnapsackModel.from_knapsack(model)
    solution = model.get_dummy_solution()
    mixed_mutation = create_mutations_portfolio_from_problem(
        problem=model, selected_mutations={BitFlipKnapsackMutation}
    )
    res = RestartHandlerLimit(3000)
    params_objective_function = get_default_objective_setup(model)
    sa = HillClimberPareto(
        problem=model,
        mutator=mixed_mutation,
        restart_handler=res,
        mode_mutation=ModeMutation.MUTATE,
        params_objective_function=params_objective_function,
        store_solution=True,
    )
    result_sa = sa.solve(
        initial_variable=solution,
        nb_iteration_max=50000,
        update_iteration_pareto=1000,
        callbacks=[TimerStopper(total_seconds=20, check_nb_steps=1000)],
    )
    pareto = result_sa
    pareto.len_pareto_front()

    # test warm start
    start_solution = result_sa[-1][0]
    sa = HillClimberPareto(
        problem=model,
        mutator=mixed_mutation,
        restart_handler=res,
        mode_mutation=ModeMutation.MUTATE,
        params_objective_function=params_objective_function,
        store_solution=True,
    )
    sa.set_warm_start(start_solution)
    result_sa2 = sa.solve(
        nb_iteration_max=5,
        update_iteration_pareto=1000,
    )
    assert result_sa[0][0].list_taken != start_solution.list_taken
    assert result_sa2[0][0].list_taken == start_solution.list_taken


if __name__ == "__main__":
    test_hc_knapsack_multiobj()
