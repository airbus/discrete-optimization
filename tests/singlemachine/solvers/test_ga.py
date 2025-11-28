#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import numpy as np

from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.generic_tools.ea.ga import Ga
from discrete_optimization.generic_tools.mutations.mutation_catalog import (
    get_available_mutations,
)
from discrete_optimization.generic_tools.mutations.mutation_portfolio import (
    create_mutations_portfolio_from_problem,
)


def test_ga(problem):
    params = get_default_objective_setup(problem)
    mixed_mutation = create_mutations_portfolio_from_problem(problem=problem)
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
