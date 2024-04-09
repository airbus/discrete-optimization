#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

import numpy as np

from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.generic_tools.do_problem import (
    BaseMethodAggregating,
    MethodAggregating,
)
from discrete_optimization.generic_tools.lns_cp import LNS_CP
from discrete_optimization.generic_tools.lns_mip import TrivialInitialSolution
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
from discrete_optimization.generic_tools.result_storage.result_storage import (
    from_solutions_to_result_storage,
    plot_fitness,
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
from discrete_optimization.knapsack.solvers.cp_solvers import (
    CPMultidimensionalMultiScenarioSolver,
    CPMultidimensionalSolver,
    KnapConstraintHandler,
)

logging.basicConfig(level=logging.DEBUG)


def run_cp_multidimensional():
    one_file = get_data_available()[10]
    knapsack_model: KnapsackModel = parse_file(one_file)
    multidimensional_knapsack = from_kp_to_multi(knapsack_model)
    cp_solver = CPMultidimensionalSolver(problem=multidimensional_knapsack)
    cp_solver.init_model(output_type=True)
    params_cp = ParametersCP.default()
    params_cp.time_limit = 5
    cp_solver.solve(parameters_cp=params_cp)


def run_ls(multiscenario_model):
    solution = multiscenario_model.get_dummy_solution()
    _, list_mutation = get_available_mutations(multiscenario_model, solution)
    list_mutation = [
        mutate[0].build(multiscenario_model, solution, **mutate[1])
        for mutate in list_mutation
    ]
    mixed_mutation = BasicPortfolioMutation(
        list_mutation, np.ones((len(list_mutation)))
    )
    res = RestartHandlerLimit(3000)
    sa = SimulatedAnnealing(
        problem=multiscenario_model,
        mutator=mixed_mutation,
        restart_handler=res,
        temperature_handler=TemperatureSchedulingFactor(1000, res, 0.99),
        mode_mutation=ModeMutation.MUTATE,
    )
    return sa.solve(
        initial_variable=solution,
        nb_iteration_max=3000,
    )


def run_cp_multidimensional_multiscenario():
    one_file = get_data_available()[10]
    knapsack_model: KnapsackModel = parse_file(one_file)
    multidimensional_knapsack = from_kp_to_multi(knapsack_model)
    scenarios = create_noised_scenario(multidimensional_knapsack, nb_scenarios=20)
    for s in scenarios:
        s.force_recompute_values = True
    multiscenario_model = MultiScenarioMultidimensionalKnapsack(
        list_problem=scenarios,
        method_aggregating=MethodAggregating(
            base_method_aggregating=BaseMethodAggregating.MEAN
        ),
    )
    solver = CPMultidimensionalMultiScenarioSolver(problem=multiscenario_model)
    solver.init_model(output_type=True)

    dummy_solution = multiscenario_model.get_dummy_solution()
    res_storage = from_solutions_to_result_storage(
        [dummy_solution], problem=multiscenario_model
    )
    lns = LNS_CP(
        problem=multiscenario_model,
        cp_solver=solver,
        initial_solution_provider=TrivialInitialSolution(res_storage),
        constraint_handler=KnapConstraintHandler(fraction_fix=0.93),
    )
    p = ParametersCP.default()
    p.time_limit = 5
    r_lns = lns.solve_lns(
        parameters_cp=p,
        nb_iteration_lns=100,
        nb_iteration_no_improvement=1000,
        callbacks=[
            TimerStopper(total_seconds=30),
        ],
    )
    plot_fitness(r_lns, title="LNS results")
    print(r_lns.get_best_solution_fit()[1])
    r_ls = run_ls(multiscenario_model=multiscenario_model)
    print(r_ls.get_best_solution_fit()[1])
    plot_fitness(r_ls, title="Local search results")


if __name__ == "__main__":
    run_cp_multidimensional()
    run_cp_multidimensional_multiscenario()
