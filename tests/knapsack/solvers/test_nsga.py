#  Copyright (c) 2022-2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveHandling,
    ParamsObjectiveFunction,
)
from discrete_optimization.generic_tools.ea.ga import DeapMutation
from discrete_optimization.generic_tools.ea.nsga import Nsga
from discrete_optimization.generic_tools.mutations.mutation_portfolio import (
    PortfolioMutation,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    plot_storage_2d,
)
from discrete_optimization.knapsack.mutation import (
    BitFlipKnapsackMutation,
    SingleBitFlipKnapsackMutation,
)
from discrete_optimization.knapsack.parser import get_data_available, parse_file


def testing_nsga_1():
    files = [f for f in get_data_available() if "ks_60_0" in f]
    knapsack_problem = parse_file(files[0])
    objectives = ["value", "weight_violation"]
    params_objective_function = ParamsObjectiveFunction(
        objective_handling=ObjectiveHandling.MULTI_OBJ,
        objectives=objectives,
        weights=[1, 1],
        sense_function=ModeOptim.MINIMIZATION,
    )
    ga_solver = Nsga(
        knapsack_problem,
        encoding="list_taken",
        params_objective_function=params_objective_function,
        mutation=DeapMutation.MUT_FLIP_BIT,
        max_evals=3000,
    )

    result_storage = ga_solver.solve()
    plot_storage_2d(result_storage=result_storage, name_axis=objectives)


def testing_own_bitflip_kp_mutation():
    files = [f for f in get_data_available() if "ks_60_0" in f]
    knapsack_problem = parse_file(files[0])
    mutation_1 = SingleBitFlipKnapsackMutation(knapsack_problem)
    mutation_2 = BitFlipKnapsackMutation(knapsack_problem)
    mutation = PortfolioMutation(
        problem=knapsack_problem,
        list_mutations=[mutation_1, mutation_2],
        weight_mutations=[0.001, 0.5],
    )
    objectives = ["value", "weight_violation"]
    params_objective_function = ParamsObjectiveFunction(
        objective_handling=ObjectiveHandling.MULTI_OBJ,
        objectives=objectives,
        weights=[1, 1],
        sense_function=ModeOptim.MINIMIZATION,
    )

    ga_solver = Nsga(
        knapsack_problem,
        encoding="list_taken",
        params_objective_function=params_objective_function,
        mutation=mutation,
        max_evals=3000,
    )
    result_storage = ga_solver.solve()
    plot_storage_2d(result_storage=result_storage, name_axis=objectives)


if __name__ == "__main__":
    testing_own_bitflip_kp_mutation()
