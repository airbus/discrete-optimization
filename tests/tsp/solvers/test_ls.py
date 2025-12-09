#  Copyright (c) 2022-2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.generic_tools.ls.hill_climber import HillClimber
from discrete_optimization.generic_tools.ls.local_search import RestartHandlerLimit
from discrete_optimization.generic_tools.ls.simulated_annealing import (
    ModeMutation,
    SimulatedAnnealing,
    TemperatureSchedulingFactor,
)
from discrete_optimization.generic_tools.mutations.mutation_catalog import (
    PartialShuffleMutation,
    SwapMutation,
    TwoOptMutation,
)
from discrete_optimization.generic_tools.mutations.mutation_portfolio import (
    create_mutations_portfolio_from_problem,
)
from discrete_optimization.tsp.mutation import SwapTspMutation, TwoOptTspMutation
from discrete_optimization.tsp.parser import get_data_available, parse_file
from discrete_optimization.tsp.problem import TspSolution


def test_sa_2opt():
    files = get_data_available()
    files = [f for f in files if "tsp_100_3" in f]
    model = parse_file(files[0])
    params_objective_function = get_default_objective_setup(problem=model)
    solution = model.get_dummy_solution()
    res = RestartHandlerLimit(3000)
    mutate_portfolio = create_mutations_portfolio_from_problem(
        problem=model, selected_mutations={SwapTspMutation, TwoOptTspMutation}
    )
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
    res = RestartHandlerLimit(3000)
    mutate_portfolio = create_mutations_portfolio_from_problem(
        problem=model, selected_mutations={PartialShuffleMutation}
    )
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
    res = RestartHandlerLimit(3000)
    mutate_portfolio = create_mutations_portfolio_from_problem(
        problem=model, selected_mutations={SwapMutation}
    )
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
    res = RestartHandlerLimit(3000)
    mutate_portfolio = create_mutations_portfolio_from_problem(
        problem=model, selected_mutations={TwoOptMutation}
    )
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
    res = RestartHandlerLimit(100)
    mutate_portfolio = create_mutations_portfolio_from_problem(
        problem=model, selected_mutations={SwapTspMutation, TwoOptTspMutation}
    )
    sa = HillClimber(
        problem=model,
        mutator=mutate_portfolio,
        restart_handler=res,
        mode_mutation=ModeMutation.MUTATE_AND_EVALUATE,
        params_objective_function=params_objective_function,
    )
    result_storage = sa.solve(initial_variable=solution, nb_iteration_max=10000)
    sol: TspSolution = result_storage.get_best_solution()
    assert model.satisfy(sol)

    # test warm start
    start_solution = sol
    assert result_storage[0][0].permutation != sol.permutation
    sa = HillClimber(
        problem=model,
        mutator=mutate_portfolio,
        restart_handler=res,
        mode_mutation=ModeMutation.MUTATE_AND_EVALUATE,
        params_objective_function=params_objective_function,
    )
    sa.set_warm_start(start_solution)
    result_storage2 = sa.solve(nb_iteration_max=10000)
    assert result_storage2[0][0].permutation == sol.permutation


if __name__ == "__main__":
    test_sa_partial_shuffle()
