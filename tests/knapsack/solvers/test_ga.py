#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.generic_tools.ea.ga import Ga
from discrete_optimization.generic_tools.mutations.mixed_mutation import (
    BasicPortfolioMutation,
)
from discrete_optimization.knapsack.mutation import (
    KnapsackMutationSingleBitFlip,
    MutationKnapsack,
)
from discrete_optimization.knapsack.parser import get_data_available, parse_file


def test_ga():
    files = [f for f in get_data_available() if "ks_60_0" in f]
    knapsack_problem = parse_file(files[0])
    params = get_default_objective_setup(knapsack_problem)
    ga_solver = Ga(
        knapsack_problem,
        objective_handling=params.objective_handling,
        objectives=params.objectives,
        objective_weights=params.weights,
    )
    results = ga_solver.solve()
    sol, fit = results.get_best_solution_fit()
    knapsack_problem.evaluate(sol)


def test_own_bitflip_kp_mutation():
    files = [f for f in get_data_available() if "ks_60_0" in f]
    knapsack_problem = parse_file(files[0])
    mutation_1 = KnapsackMutationSingleBitFlip(knapsack_problem)
    mutation_2 = MutationKnapsack(knapsack_problem)
    mutation = BasicPortfolioMutation(
        [mutation_1, mutation_2], weight_mutation=[0.001, 0.5]
    )
    params_objective_function = get_default_objective_setup(knapsack_problem)
    ga_solver = Ga(
        knapsack_problem,
        objective_handling=params_objective_function.objective_handling,
        objectives=params_objective_function.objectives,
        objective_weights=params_objective_function.weights,
        mutation=mutation,
        max_evals=3000,
    )
    results = ga_solver.solve()
    sol, fit = results.get_best_solution_fit()
    knapsack_problem.evaluate(sol)


if __name__ == "__main__":
    test_own_bitflip_kp_mutation()
