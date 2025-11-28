#  Copyright (c) 2022-2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

from discrete_optimization.generic_tools.do_problem import (
    BaseMethodAggregating,
    MethodAggregating,
)
from discrete_optimization.generic_tools.ls.local_search import (
    ModeMutation,
    RestartHandlerLimit,
)
from discrete_optimization.generic_tools.ls.simulated_annealing import (
    SimulatedAnnealing,
    TemperatureSchedulingFactor,
)
from discrete_optimization.generic_tools.mutations.mutation_portfolio import (
    create_mutations_portfolio_from_problem,
)
from discrete_optimization.knapsack.parser import get_data_available, parse_file
from discrete_optimization.knapsack.problem import (
    KnapsackProblem,
    MultiScenarioMultidimensionalKnapsackProblem,
    create_noised_scenario,
    from_kp_to_multi,
)


def initialize_multiscenario():
    one_file = get_data_available()[10]
    knapsack_problem: KnapsackProblem = parse_file(one_file)
    multidimensional_knapsack = from_kp_to_multi(knapsack_problem)
    scenarios = create_noised_scenario(multidimensional_knapsack, nb_scenarios=20)
    for s in scenarios:
        s.force_recompute_values = True
    multiscenario_model = MultiScenarioMultidimensionalKnapsackProblem(
        list_problem=scenarios,
        method_aggregating=MethodAggregating(
            base_method_aggregating=BaseMethodAggregating.MEAN
        ),
    )
    solution = multiscenario_model.get_dummy_solution()
    mixed_mutation = create_mutations_portfolio_from_problem(
        problem=multiscenario_model,
    )
    res = RestartHandlerLimit(3000)
    sa = SimulatedAnnealing(
        problem=multiscenario_model,
        mutator=mixed_mutation,
        restart_handler=res,
        temperature_handler=TemperatureSchedulingFactor(1000, res, 0.99),
        mode_mutation=ModeMutation.MUTATE,
    )

    logging.basicConfig(level=logging.DEBUG)
    res = sa.solve(
        initial_variable=solution,
        nb_iteration_max=5000,
    )


if __name__ == "__main__":
    initialize_multiscenario()
