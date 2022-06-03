import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../"))

import matplotlib.pyplot as plt
from discrete_optimization.generic_tools.ea.ga import DeapMutation
from discrete_optimization.generic_tools.ea.nsga import Nsga
from discrete_optimization.generic_tools.mutations.mixed_mutation import (
    BasicPortfolioMutation,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    plot_storage_2d,
)
from discrete_optimization.knapsack.knapsack_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.knapsack.mutation.mutation_knapsack import (
    KnapsackMutationSingleBitFlip,
    MutationKnapsack,
)


def testing_nsga_1():
    files = [f for f in get_data_available() if "ks_60_0" in f]
    knapsack_model = parse_file(files[0])
    objectives = ["value", "weight_violation"]
    ga_solver = Nsga(
        knapsack_model,
        encoding="list_taken",
        objectives=objectives,
        objective_weights=[-1, -1],
        mutation=DeapMutation.MUT_FLIP_BIT,
        max_evals=3000,
    )

    result_storage = ga_solver.solve()
    print(result_storage)

    # pareto_front = ParetoFront(result_storage)
    # print('pareto_front: ', pareto_front)

    # plot_pareto_2d(result_storage, name_axis=objectives)
    plot_storage_2d(result_storage=result_storage, name_axis=objectives)


def testing_own_bitflip_kp_mutation():
    files = [f for f in get_data_available() if "ks_60_0" in f]
    knapsack_model = parse_file(files[0])
    mutation_1 = KnapsackMutationSingleBitFlip(knapsack_model)
    mutation_2 = MutationKnapsack(knapsack_model)
    mutation = BasicPortfolioMutation(
        [mutation_1, mutation_2], weight_mutation=[0.001, 0.5]
    )
    objectives = ["value", "weight_violation"]

    print("objectives", objectives)
    # objectives = ['value', 'weight_violation']
    ga_solver = Nsga(
        knapsack_model,
        encoding="list_taken",
        objectives=objectives,
        objective_weights=[-1, -1],
        mutation=mutation,
        # mutation=mutation_1,
        # mutation=DeapMutation.MUT_FLIP_BIT,
        max_evals=3000,
    )
    result_storage = ga_solver.solve()
    print(result_storage)

    # pareto_front = ParetoFront(result_storage)
    # print('pareto_front: ', pareto_front)

    # plot_pareto_2d(result_storage, name_axis=objectives)
    plot_storage_2d(result_storage=result_storage, name_axis=objectives)


if __name__ == "__main__":
    # testing_nsga_1()
    testing_own_bitflip_kp_mutation()
