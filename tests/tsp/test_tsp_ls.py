#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import numpy as np

from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.generic_tools.ls.hill_climber import HillClimber
from discrete_optimization.generic_tools.ls.local_search import RestartHandlerLimit
from discrete_optimization.generic_tools.ls.simulated_annealing import (
    ModeMutation,
    SimulatedAnnealing,
    TemperatureSchedulingFactor,
)
from discrete_optimization.generic_tools.mutations.mixed_mutation import (
    BasicPortfolioMutation,
)
from discrete_optimization.generic_tools.mutations.mutation_catalog import (
    PermutationPartialShuffleMutation,
    PermutationSwap,
    TwoOptMutation,
    get_available_mutations,
)
from discrete_optimization.tsp.mutation.mutation_tsp import (
    Mutation2Opt,
    MutationSwapTSP,
)
from discrete_optimization.tsp.tsp_parser import get_data_available, parse_file


def test_sa_2opt():
    files = get_data_available()
    files = [f for f in files if "tsp_100_3" in f]
    model = parse_file(files[0])
    params_objective_function = get_default_objective_setup(problem=model)
    solution = model.get_dummy_solution()
    _, list_mutation = get_available_mutations(model, solution)
    res = RestartHandlerLimit(3000)
    list_mutation = [
        mutate[0].build(model, solution, attribute="permutation", **mutate[1])
        for mutate in list_mutation
        if mutate[0] in [MutationSwapTSP, Mutation2Opt]
    ]
    weight = np.ones(len(list_mutation))
    mutate_portfolio = BasicPortfolioMutation(list_mutation, weight)
    sa = SimulatedAnnealing(
        problem=model,
        mutator=mutate_portfolio,
        restart_handler=res,
        temperature_handler=TemperatureSchedulingFactor(
            temperature=100, restart_handler=res, coefficient=0.99999
        ),
        mode_mutation=ModeMutation.MUTATE_AND_EVALUATE,
        params_objective_function=params_objective_function,
    )
    sol = sa.solve(
        initial_variable=solution, nb_iteration_max=10000
    ).get_best_solution()
    assert model.satisfy(sol)


def test_sa_partial_shuffle():
    files = get_data_available()
    files = [f for f in files if "tsp_100_3" in f]
    model = parse_file(files[0])
    params_objective_function = get_default_objective_setup(problem=model)
    solution = model.get_dummy_solution()
    _, list_mutation = get_available_mutations(model, solution)
    res = RestartHandlerLimit(3000)
    list_mutation = [
        mutate[0].build(model, solution, attribute="permutation", **mutate[1])
        for mutate in list_mutation
        if mutate[0] in [PermutationPartialShuffleMutation]
    ]
    weight = np.ones(len(list_mutation))
    mutate_portfolio = BasicPortfolioMutation(list_mutation, weight)
    sa = SimulatedAnnealing(
        problem=model,
        mutator=mutate_portfolio,
        restart_handler=res,
        temperature_handler=TemperatureSchedulingFactor(
            temperature=100, restart_handler=res, coefficient=0.99999
        ),
        mode_mutation=ModeMutation.MUTATE_AND_EVALUATE,
        params_objective_function=params_objective_function,
    )
    sol = sa.solve(
        initial_variable=solution, nb_iteration_max=10000
    ).get_best_solution()


def test_sa_swap():
    files = get_data_available()
    files = [f for f in files if "tsp_100_3" in f]
    model = parse_file(files[0])
    params_objective_function = get_default_objective_setup(problem=model)
    solution = model.get_dummy_solution()
    _, list_mutation = get_available_mutations(model, solution)
    res = RestartHandlerLimit(3000)
    list_mutation = [
        mutate[0].build(model, solution, attribute="permutation", **mutate[1])
        for mutate in list_mutation
        if mutate[0] in [PermutationSwap]
    ]
    weight = np.ones(len(list_mutation))
    mutate_portfolio = BasicPortfolioMutation(list_mutation, weight)
    sa = SimulatedAnnealing(
        problem=model,
        mutator=mutate_portfolio,
        restart_handler=res,
        temperature_handler=TemperatureSchedulingFactor(
            temperature=100, restart_handler=res, coefficient=0.99999
        ),
        mode_mutation=ModeMutation.MUTATE_AND_EVALUATE,
        params_objective_function=params_objective_function,
    )
    sol = sa.solve(
        initial_variable=solution, nb_iteration_max=10000
    ).get_best_solution()
    assert model.satisfy(sol)


def test_sa_twoopttbasic():
    files = get_data_available()
    files = [f for f in files if "tsp_100_3" in f]
    model = parse_file(files[0])
    params_objective_function = get_default_objective_setup(problem=model)
    solution = model.get_dummy_solution()
    _, list_mutation = get_available_mutations(model, solution)
    res = RestartHandlerLimit(3000)
    list_mutation = [
        mutate[0].build(model, solution, attribute="permutation", **mutate[1])
        for mutate in list_mutation
        if mutate[0] in [TwoOptMutation]
    ]
    weight = np.ones(len(list_mutation))
    mutate_portfolio = BasicPortfolioMutation(list_mutation, weight)
    sa = SimulatedAnnealing(
        problem=model,
        mutator=mutate_portfolio,
        restart_handler=res,
        temperature_handler=TemperatureSchedulingFactor(
            temperature=100, restart_handler=res, coefficient=0.99999
        ),
        mode_mutation=ModeMutation.MUTATE_AND_EVALUATE,
        params_objective_function=params_objective_function,
    )
    sol = sa.solve(
        initial_variable=solution, nb_iteration_max=10000
    ).get_best_solution()
    assert model.satisfy(sol)


def test_hc():
    files = get_data_available()
    files = [f for f in files if "tsp_100_3" in f]
    model = parse_file(files[0])
    params_objective_function = get_default_objective_setup(problem=model)
    solution = model.get_dummy_solution()
    _, list_mutation = get_available_mutations(model, solution)
    res = RestartHandlerLimit(100)
    list_mutation = [
        mutate[0].build(model, solution, **mutate[1])
        for mutate in list_mutation
        if mutate[0] in [MutationSwapTSP, Mutation2Opt]
    ]
    weight = np.ones(len(list_mutation))
    mutate_portfolio = BasicPortfolioMutation(list_mutation, weight)
    sa = HillClimber(
        problem=model,
        mutator=mutate_portfolio,
        restart_handler=res,
        mode_mutation=ModeMutation.MUTATE_AND_EVALUATE,
        params_objective_function=params_objective_function,
    )
    sol = sa.solve(
        initial_variable=solution, nb_iteration_max=10000
    ).get_best_solution()
    assert model.satisfy(sol)


if __name__ == "__main__":
    test_sa_partial_shuffle()
