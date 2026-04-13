#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Test local search solvers for single batch processing."""

from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.ls.hill_climber import HillClimber
from discrete_optimization.generic_tools.ls.local_search import RestartHandlerLimit
from discrete_optimization.generic_tools.ls.simulated_annealing import (
    ModeMutation,
    SimulatedAnnealing,
    TemperatureSchedulingFactor,
)
from discrete_optimization.generic_tools.mutations.mutation_portfolio import (
    create_mutations_portfolio_from_problem,
)


def test_sa_singlebatch(tiny_problem):
    """Test simulated annealing on single batch processing."""
    solution = tiny_problem.get_dummy_solution()
    mixed_mutation = create_mutations_portfolio_from_problem(problem=tiny_problem)
    res = RestartHandlerLimit(1000)
    sa = SimulatedAnnealing(
        problem=tiny_problem,
        mutator=mixed_mutation,
        restart_handler=res,
        temperature_handler=TemperatureSchedulingFactor(100, res, 0.95),
        mode_mutation=ModeMutation.MUTATE,
    )
    result = sa.solve(
        initial_variable=solution,
        nb_iteration_max=10,
        callbacks=[TimerStopper(total_seconds=5, check_nb_steps=5)],
    )
    sol, fit = result.get_best_solution_fit()
    assert tiny_problem.satisfy(sol)


def test_hc_singlebatch(tiny_problem):
    """Test hill climber on single batch processing."""
    solution = tiny_problem.get_dummy_solution()
    mixed_mutation = create_mutations_portfolio_from_problem(problem=tiny_problem)
    res = RestartHandlerLimit(1000)
    hc = HillClimber(
        problem=tiny_problem,
        mutator=mixed_mutation,
        restart_handler=res,
        mode_mutation=ModeMutation.MUTATE,
    )
    result = hc.solve(
        initial_variable=solution,
        nb_iteration_max=10,
        callbacks=[TimerStopper(total_seconds=5, check_nb_steps=5)],
    )
    sol, fit = result.get_best_solution_fit()
    assert tiny_problem.satisfy(sol)
