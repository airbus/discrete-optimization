#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import numpy as np

from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.generic_tools.ea.ga import Ga
from discrete_optimization.generic_tools.mutations.mixed_mutation import (
    BasicPortfolioMutation,
)
from discrete_optimization.generic_tools.mutations.mutation_catalog import (
    get_available_mutations,
)


def test_ga(problem):
    dummy = problem.get_dummy_solution()
    params = get_default_objective_setup(problem)
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
        max_evals=100,
        mut_rate=0.1,
        crossover_rate=0.9,
        tournament_size=0.2,  #
    )
    results = ga_solver.solve()
    sol, fit = results.get_best_solution_fit()
    assert problem.satisfy(sol)
