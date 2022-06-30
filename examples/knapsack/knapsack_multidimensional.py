import numpy as np

from discrete_optimization.generic_tools.do_problem import (
    BaseMethodAggregating,
    MethodAggregating,
)
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
    LNS_CP,
    CPMultidimensionalMultiScenarioSolver,
    CPMultidimensionalSolver,
    KnapConstraintHandler,
    ParametersCP,
)


def test_cp_multidimensional():
    one_file = get_data_available()[10]
    knapsack_model: KnapsackModel = parse_file(one_file)
    multidimensional_knapsack = from_kp_to_multi(knapsack_model)
    cp_solver = CPMultidimensionalSolver(knapsack_model=multidimensional_knapsack)
    cp_solver.init_model(output_type=True)
    cp_solver.solve(parameters_cp=ParametersCP.default())


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
    res = RestartHandlerLimit(3000, solution, multiscenario_model.evaluate(solution))
    sa = SimulatedAnnealing(
        evaluator=multiscenario_model,
        mutator=mixed_mutation,
        restart_handler=res,
        temperature_handler=TemperatureSchedulingFactor(1000, res, 0.99),
        mode_mutation=ModeMutation.MUTATE,
    )
    return sa.solve(
        initial_variable=solution,
        nb_iteration_max=60000,
        pickle_result=False,
        verbose=False,
    )


def test_cp_multidimensional_multiscenario():
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
    solver = CPMultidimensionalMultiScenarioSolver(knapsack_model=multiscenario_model)
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
    p.TimeLimit = 5
    r_lns = lns.solve_lns(
        parameters_cp=p,
        nb_iteration_lns=1000,
        nb_iteration_no_improvement=1000,
        max_time_seconds=140,
    )
    plot_fitness(r_lns, title="LNS results")
    print(r_lns.get_best_solution_fit()[1])
    # r = solver.solve(parameters_cp=ParametersCP.default())
    # print(r.get_best_solution_fit()[1])
    r_ls = run_ls(multiscenario_model=multiscenario_model)
    print(r_ls.get_best_solution_fit()[1])
    plot_fitness(r_ls, title="Local search results")


if __name__ == "__main__":
    test_cp_multidimensional_multiscenario()
