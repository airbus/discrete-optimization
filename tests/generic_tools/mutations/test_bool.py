#  Copyright (c) 2022-2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tools.mutations.mutation_bool import BitFlipMutation
from discrete_optimization.knapsack.mutation import BitFlipKnapsackMutation
from discrete_optimization.knapsack.parser import get_data_available, parse_file
from discrete_optimization.knapsack.problem import KnapsackProblem


def testing_on_knapsack():
    model_file = [f for f in get_data_available() if "ks_60_0" in f][0]
    model: KnapsackProblem = parse_file(model_file, force_recompute_values=True)
    solution = model.get_dummy_solution()
    mutation = BitFlipMutation(model)
    mutation_2 = BitFlipKnapsackMutation(model)
    for i in range(1000):
        sol, move, f = mutation_2.mutate_and_compute_obj(solution)
    sol_back = move.backtrack_local_move(sol)
    f = model.evaluate(sol_back)


if __name__ == "__main__":
    testing_on_knapsack()
