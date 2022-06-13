import os
import sys

import numpy as np
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    get_default_objective_setup,
)
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
    # mutation = Mutation2Opt(model, False, 100, False)
    solution = model.get_dummy_solution()
    _, list_mutation = get_available_mutations(model, solution)
    res = RestartHandlerLimit(3000, solution, model.evaluate(solution))
    print(list_mutation)
    list_mutation = [
        mutate[0].build(model, solution, attribute="permutation", **mutate[1])
        for mutate in list_mutation
        if mutate[0] in [MutationSwapTSP, Mutation2Opt]
    ]
    weight = np.ones(len(list_mutation))
    mutate_portfolio = BasicPortfolioMutation(list_mutation, weight)
    sa = SimulatedAnnealing(
        evaluator=model,
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
    # assert model.satisfy(sol)


def test_sa_partial_shuffle():
    files = get_data_available()
    files = [f for f in files if "tsp_100_3" in f]
    model = parse_file(files[0])
    params_objective_function = get_default_objective_setup(problem=model)
    # mutation = Mutation2Opt(model, False, 100, False)
    solution = model.get_dummy_solution()
    _, list_mutation = get_available_mutations(model, solution)
    res = RestartHandlerLimit(3000, solution, model.evaluate(solution))
    print(list_mutation)
    list_mutation = [
        mutate[0].build(model, solution, attribute="permutation", **mutate[1])
        for mutate in list_mutation
        if mutate[0] in [PermutationPartialShuffleMutation]
    ]
    weight = np.ones(len(list_mutation))
    mutate_portfolio = BasicPortfolioMutation(list_mutation, weight)
    sa = SimulatedAnnealing(
        evaluator=model,
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
    # assert model.satisfy(sol)


def test_sa_swap():
    files = get_data_available()
    files = [f for f in files if "tsp_100_3" in f]
    model = parse_file(files[0])
    params_objective_function = get_default_objective_setup(problem=model)
    # mutation = Mutation2Opt(model, False, 100, False)
    solution = model.get_dummy_solution()
    _, list_mutation = get_available_mutations(model, solution)
    res = RestartHandlerLimit(3000, solution, model.evaluate(solution)["length"])
    list_mutation = [
        mutate[0].build(model, solution, attribute="permutation", **mutate[1])
        for mutate in list_mutation
        if mutate[0] in [PermutationSwap]
    ]
    weight = np.ones(len(list_mutation))
    mutate_portfolio = BasicPortfolioMutation(list_mutation, weight)
    sa = SimulatedAnnealing(
        evaluator=model,
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
    # assert model.satisfy(sol)


def test_sa_twoopttbasic():
    files = get_data_available()
    files = [f for f in files if "tsp_100_3" in f]
    model = parse_file(files[0])
    params_objective_function = get_default_objective_setup(problem=model)
    solution = model.get_dummy_solution()
    _, list_mutation = get_available_mutations(model, solution)
    res = RestartHandlerLimit(3000, solution, model.evaluate(solution))
    print(list_mutation)
    list_mutation = [
        mutate[0].build(model, solution, attribute="permutation", **mutate[1])
        for mutate in list_mutation
        if mutate[0] in [TwoOptMutation]
    ]
    print(list_mutation[0].attribute)
    weight = np.ones(len(list_mutation))
    mutate_portfolio = BasicPortfolioMutation(list_mutation, weight)
    sa = SimulatedAnnealing(
        evaluator=model,
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
    # assert model.satisfy(sol)


def test_hc():
    files = get_data_available()
    files = [f for f in files if "tsp_100_3" in f]
    model = parse_file(files[0])
    params_objective_function = get_default_objective_setup(problem=model)
    # mutation = Mutation2Opt(model, False, 100, False)
    solution = model.get_dummy_solution()
    _, list_mutation = get_available_mutations(model, solution)
    res = RestartHandlerLimit(100, solution, model.evaluate(solution))
    print(list_mutation)
    list_mutation = [
        mutate[0].build(model, solution, **mutate[1])
        for mutate in list_mutation
        if mutate[0] in [MutationSwapTSP, Mutation2Opt]
    ]
    weight = np.ones(len(list_mutation))
    mutate_portfolio = BasicPortfolioMutation(list_mutation, weight)
    sa = HillClimber(
        evaluator=model,
        mutator=mutate_portfolio,
        restart_handler=res,
        mode_mutation=ModeMutation.MUTATE_AND_EVALUATE,
        params_objective_function=params_objective_function,
    )
    sol = sa.solve(
        initial_variable=solution, nb_iteration_max=10000
    ).get_best_solution()
    # assert model.satisfy(sol)


if __name__ == "__main__":
    test_sa_partial_shuffle()
