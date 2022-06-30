import numpy as np

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
from discrete_optimization.generic_tools.mutations.mixed_mutation import (
    BasicPortfolioMutation,
)
from discrete_optimization.generic_tools.mutations.mutation_catalog import (
    get_available_mutations,
)
from discrete_optimization.knapsack.knapsack_model import (
    KnapsackModel,
    MultiScenarioMultidimensionalKnapsack,
    create_noised_scenario,
    from_kp_to_multi,
)
from discrete_optimization.knapsack.knapsack_parser import (
    get_data_available,
    parse_file,
)


def initialize_multiscenario():
    one_file = get_data_available()[10]
    knapsack_model: KnapsackModel = parse_file(one_file)
    multidimensional_knapsack = from_kp_to_multi(knapsack_model)
    scenarios = create_noised_scenario(multidimensional_knapsack, nb_scenarios=20)
    # scenarios = [from_kp_to_multi(parse_file(files_available[0]))
    #              for i in range(10)]
    for s in scenarios:
        s.force_recompute_values = True
    multiscenario_model = MultiScenarioMultidimensionalKnapsack(
        list_problem=scenarios,
        method_aggregating=MethodAggregating(
            base_method_aggregating=BaseMethodAggregating.MEAN
        ),
    )
    solution = multiscenario_model.get_dummy_solution()
    _, list_mutation = get_available_mutations(multiscenario_model, solution)
    list_mutation = [
        mutate[0].build(multiscenario_model, solution, **mutate[1])
        for mutate in list_mutation
    ]
    mixed_mutation = BasicPortfolioMutation(
        list_mutation, np.ones((len(list_mutation)))
    )
    res = RestartHandlerLimit(3000, solution, multiscenario_model.evaluate(solution))
    sa = SimulatedAnnealing(
        evaluator=multiscenario_model,
        mutator=mixed_mutation,
        restart_handler=res,
        temperature_handler=TemperatureSchedulingFactor(1000, res, 0.99),
        mode_mutation=ModeMutation.MUTATE,
    )
    sa.solve(
        initial_variable=solution,
        nb_iteration_max=1000000,
        pickle_result=False,
        verbose=False,
    )


if __name__ == "__main__":
    initialize_multiscenario()
