#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tools.mutations.mutation_bool import MutationBitFlip
from discrete_optimization.knapsack.knapsack_model import KnapsackModel
from discrete_optimization.knapsack.knapsack_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.knapsack.mutation.mutation_knapsack import MutationKnapsack


def testing_on_knapsack():
    model_file = [f for f in get_data_available() if "ks_60_0" in f][0]
    model: KnapsackModel = parse_file(model_file, force_recompute_values=True)
    solution = model.get_dummy_solution()
    mutation = MutationBitFlip(model)
    mutation_2 = MutationKnapsack(model)
    for i in range(1000):
        sol, move, f = mutation_2.mutate_and_compute_obj(solution)
    sol_back = move.backtrack_local_move(sol)
    f = model.evaluate(sol_back)


if __name__ == "__main__":
    testing_on_knapsack()
