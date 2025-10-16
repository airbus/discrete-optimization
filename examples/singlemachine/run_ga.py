#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

import numpy as np
from didppy import BeamParallelizationMethod

from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.generic_tools.ea.ga import Ga
from discrete_optimization.generic_tools.ls.local_search import (
    ModeMutation,
    RestartHandlerLimit,
)
from discrete_optimization.generic_tools.ls.simulated_annealing import (
    TemperatureSchedulingFactor,
)
from discrete_optimization.generic_tools.mutations.mixed_mutation import (
    BasicPortfolioMutation,
)
from discrete_optimization.generic_tools.mutations.mutation_catalog import (
    get_available_mutations,
)
from discrete_optimization.generic_tools.mutations.permutation_mutations import (
    PermutationPartialShuffleMutation,
    PermutationShuffleMutation,
    ShuffleMove,
    TwoOptMutation,
)
from discrete_optimization.knapsack.mutation import (
    KnapsackMutationSingleBitFlip,
    MutationKnapsack,
)
from discrete_optimization.singlemachine.parser import get_data_available, parse_file
from discrete_optimization.singlemachine.problem import WeightedTardinessProblem
from discrete_optimization.singlemachine.solvers.cpsat import CpsatWTSolver
from discrete_optimization.singlemachine.solvers.dp import DpWTSolver, dp

logging.basicConfig(level=logging.DEBUG)


def run_ga():
    problems = parse_file(get_data_available()[0])
    print(len(problems), " problems in the file")
    problem = problems[3]
    dummy = problem.get_dummy_solution()
    params = get_default_objective_setup(problem)
    _, mutations = get_available_mutations(problem)
    print(mutations)
    _, mutations = get_available_mutations(problem, dummy)
    list_mutation = [
        mutate[0].build(problem, dummy, **mutate[1]) for mutate in mutations
    ]
    mixed_mutation = BasicPortfolioMutation(
        list_mutation, np.ones((len(list_mutation)))
    )
    ga_solver = Ga(
        problem,
        encoding="permutation",
        mutation=mixed_mutation,
        objective_handling=params.objective_handling,
        objectives=params.objectives,
        objective_weights=[-w for w in params.weights],
        pop_size=10,
        max_evals=1000000,
        mut_rate=0.1,
        crossover_rate=0.9,
        tournament_size=0.2,  #
    )
    results = ga_solver.solve()
    sol, fit = results.get_best_solution_fit()
    print(problem.evaluate(sol))


def run_ls():
    from discrete_optimization.generic_tools.ls.simulated_annealing import (
        SimulatedAnnealing,
    )

    problems = parse_file(get_data_available()[0])
    print(len(problems), " problems in the file")
    problem = problems[2]
    solution = problem.get_dummy_solution()
    _, list_mutation = get_available_mutations(problem, solution)
    res = RestartHandlerLimit(3000)
    print(list_mutation)
    list_mutation = [
        mutate[0].build(problem, solution, attribute="permutation", **mutate[1])
        for mutate in [list_mutation[2]]
    ]
    weight = np.ones(len(list_mutation))
    mutate_portfolio = BasicPortfolioMutation(list_mutation, weight)
    sa = SimulatedAnnealing(
        problem=problem,
        mutator=mutate_portfolio,
        restart_handler=res,
        temperature_handler=TemperatureSchedulingFactor(
            temperature=10000, restart_handler=res, coefficient=0.99999
        ),
        mode_mutation=ModeMutation.MUTATE,
    )
    res = sa.solve(nb_iteration_max=10000000, initial_variable=solution)
    sol = res.get_best_solution()
    print(problem.evaluate(sol), problem.satisfy(sol))


if __name__ == "__main__":
    run_ga()
