import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../"))
from discrete_optimization.knapsack.knapsack_model import (
    KnapsackModel,
    KnapsackModel_Mobj,
    ObjectiveHandling,
)
from discrete_optimization.knapsack.knapsack_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.knapsack.mutation.mutation_knapsack import MutationKnapsack

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
import numpy as np
from discrete_optimization.generic_tools.do_problem import ModeOptim
from discrete_optimization.generic_tools.ls.hill_climber import HillClimberPareto
from discrete_optimization.generic_tools.ls.local_search import RestartHandlerLimit
from discrete_optimization.generic_tools.ls.simulated_annealing import (
    ModeMutation,
    SimulatedAnnealing,
    TemperatureSchedulingFactor,
)
from discrete_optimization.generic_tools.mutations.mixed_mutation import (
    BasicPortfolioMutation,
)
from discrete_optimization.generic_tools.mutations.mutation_catalog import (
    get_available_mutations,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    plot_pareto_2d,
    plot_storage_2d,
)


def sa_knapsack():
    model_file = [f for f in get_data_available() if "ks_60_0" in f][0]
    model: KnapsackModel = parse_file(model_file, force_recompute_values=True)
    solution = model.get_dummy_solution()
    _, list_mutation = get_available_mutations(model, solution)
    list_mutation = [
        mutate[0].build(model, solution, **mutate[1]) for mutate in list_mutation
    ]
    mixed_mutation = BasicPortfolioMutation(
        list_mutation, np.ones((len(list_mutation)))
    )
    res = RestartHandlerLimit(3000, solution, model.evaluate(solution))
    sa = SimulatedAnnealing(
        evaluator=model,
        mutator=mixed_mutation,
        restart_handler=res,
        temperature_handler=TemperatureSchedulingFactor(1000, res, 0.99),
        mode_mutation=ModeMutation.MUTATE,
    )
    sa.solve(solution, 1000000, max_time_seconds=100, pickle_result=False)


def hc_knapsack_multiobj():
    model_file = [f for f in get_data_available() if "ks_60_0" in f][0]
    model: KnapsackModel = parse_file(model_file, force_recompute_values=True)
    model: KnapsackModel_Mobj = KnapsackModel_Mobj.from_knapsack(model)
    solution = model.get_dummy_solution()
    _, list_mutation = get_available_mutations(model, solution)
    list_mutation = [
        mutate[0].build(model, solution, **mutate[1])
        for mutate in list_mutation
        if mutate[0] == MutationKnapsack
    ]
    mixed_mutation = BasicPortfolioMutation(
        list_mutation, np.ones((len(list_mutation)))
    )
    res = RestartHandlerLimit(3000, solution, model.evaluate(solution))
    sa = HillClimberPareto(
        evaluator=model,
        mutator=mixed_mutation,
        restart_handler=res,
        mode_mutation=ModeMutation.MUTATE,
        params_objective_function=None,
        store_solution=True,
        nb_solutions=50000,
    )
    result_sa = sa.solve(
        initial_variable=solution,
        nb_iteration_max=50000,
        max_time_seconds=100,
        update_iteration_pareto=1000,
        pickle_result=False,
    )
    pareto = result_sa.result_storage
    print(pareto.len_pareto_front())
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1)
    plot_storage_2d(result_storage=pareto, name_axis=["value", "heaviest"], ax=ax)
    plot_pareto_2d(pareto_front=pareto, name_axis=["value", "heaviest"], ax=ax)
    plt.show()


if __name__ == "__main__":
    hc_knapsack_multiobj()
