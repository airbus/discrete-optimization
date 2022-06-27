from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from discrete_optimization.generic_tools.cp_tools import CPSolverName
from discrete_optimization.generic_tools.do_problem import (
    BaseMethodAggregating,
    ModeOptim,
    ParamsObjectiveFunction,
)
from discrete_optimization.generic_tools.ls.hill_climber import (
    HillClimberPareto,
    ObjectiveHandling,
)
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
    PermutationMutationRCPSP,
    get_available_mutations,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    result_storage_to_pareto_front,
)
from discrete_optimization.rcpsp.rcpsp_model import (
    Aggreg_RCPSPModel,
    MethodAggregating,
    MethodBaseRobustification,
    MethodRobustification,
    MultiModeRCPSPModel,
    RCPSPModel,
    RCPSPSolution,
    UncertainRCPSPModel,
    create_poisson_laws_duration,
    create_poisson_laws_resource,
)
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file


def tree():
    return defaultdict(tree)


def create_models(
    base_rcpsp_model: RCPSPModel,
    range_around_mean_resource: int = 1,
    range_around_mean_duration: int = 3,
    do_uncertain_resource: bool = True,
    do_uncertain_duration: bool = True,
    nb_sampled_scenario: int = 50,
):
    poisson_laws = tree()
    if do_uncertain_duration:
        poisson_laws_duration = create_poisson_laws_duration(
            base_rcpsp_model, range_around_mean=range_around_mean_duration
        )
        for job in poisson_laws_duration:
            for mode in poisson_laws_duration[job]:
                for res in poisson_laws_duration[job][mode]:
                    poisson_laws[job][mode][res] = poisson_laws_duration[job][mode][res]
    if do_uncertain_resource:
        poisson_laws_resource = create_poisson_laws_resource(
            base_rcpsp_model, range_around_mean=range_around_mean_resource
        )
        for job in poisson_laws_resource:
            for mode in poisson_laws_resource[job]:
                for res in poisson_laws_resource[job][mode]:
                    poisson_laws[job][mode][res] = poisson_laws_resource[job][mode][res]

    uncertain = UncertainRCPSPModel(base_rcpsp_model, poisson_laws=poisson_laws)
    worst = uncertain.create_rcpsp_model(
        MethodRobustification(MethodBaseRobustification.WORST_CASE, percentile=0)
    )
    average = uncertain.create_rcpsp_model(
        MethodRobustification(MethodBaseRobustification.AVERAGE, percentile=0)
    )
    many_random_instance = [
        uncertain.create_rcpsp_model(
            method_robustification=MethodRobustification(
                MethodBaseRobustification.SAMPLE
            )
        )
        for i in range(nb_sampled_scenario)
    ]
    # many_random_instance = []
    # many_random_instance = []
    # many_random_instance += [uncertain.create_rcpsp_model(method_robustification=
    #                                                       MethodRobustification(MethodBaseRobustification.PERCENTILE,
    #                                                                            percentile=j))
    #                         for j in range(0, 100, 1)]
    return worst, average, many_random_instance


def run_cp_multiscenario():
    files = get_data_available()
    files = [f for f in files if "j301_1.sm" in f]  # Single mode RCPSP
    # files = [f for f in files if 'j601_1.sm' in f]  # Single mode RCPSP
    print(files)
    # files = [f for f in files if 'j1010_5.mm' in f]  # Multi mode RCPSP
    file_path = files[0]
    rcpsp_model = parse_file(file_path)
    poisson_laws = create_poisson_laws_duration(rcpsp_model, range_around_mean=2)
    uncertain_model: UncertainRCPSPModel = UncertainRCPSPModel(
        base_rcpsp_model=rcpsp_model, poisson_laws=poisson_laws
    )
    list_rcpsp_model = [
        uncertain_model.create_rcpsp_model(
            MethodRobustification(
                method_base=MethodBaseRobustification.SAMPLE, percentile=0
            )
        )
        for i in range(20)
    ]

    for m in list_rcpsp_model:
        m.update_functions()
    dummy = list_rcpsp_model[0].get_dummy_solution()
    model_aggreg_mean = Aggreg_RCPSPModel(
        list_problem=list_rcpsp_model,
        method_aggregating=MethodAggregating(BaseMethodAggregating.MEAN),
    )
    _, mutations = get_available_mutations(list_rcpsp_model[0], dummy)
    print(mutations)
    list_mutation = [
        mutate[0].build(rcpsp_model, dummy, **mutate[1])
        for mutate in mutations
        if mutate[0] == PermutationMutationRCPSP
    ]
    mixed_mutation = BasicPortfolioMutation(
        list_mutation, np.ones((len(list_mutation)))
    )
    res = RestartHandlerLimit(
        500, cur_solution=dummy, cur_objective=model_aggreg_mean.evaluate(dummy)
    )
    simulated_annealing = SimulatedAnnealing(
        evaluator=model_aggreg_mean,
        mutator=mixed_mutation,
        mode_mutation=ModeMutation.MUTATE,
        restart_handler=res,
        temperature_handler=TemperatureSchedulingFactor(
            temperature=1, restart_handler=res, coefficient=0.9999
        ),
        store_solution=True,
        nb_solutions=10000000,
    )
    result_sa = simulated_annealing.solve(initial_variable=dummy, nb_iteration_max=300)
    best_solution: RCPSPSolution = result_sa.get_best_solution()
    permutation = best_solution.rcpsp_permutation
    annealing = []
    for model in list_rcpsp_model:
        s = RCPSPSolution(problem=model, rcpsp_permutation=permutation)
        annealing += [model.evaluate(s)["makespan"]]
    from discrete_optimization.rcpsp.solver.cp_solvers_multiscenario import (
        CP_MULTISCENARIO,
        ParametersCP,
    )

    solver = CP_MULTISCENARIO(
        list_rcpsp_model=list_rcpsp_model, cp_solver_name=CPSolverName.CHUFFED
    )
    solver.init_model(
        output_type=True, relax_ordering=False, nb_incoherence_limit=2, max_time=300
    )
    params_cp = ParametersCP.default()
    params_cp.TimeLimit = 500
    params_cp.free_search = True
    result = solver.solve(parameters_cp=params_cp)
    solution_fit = result.list_solution_fits
    objectives_cp = [s[0][1] for s in solution_fit]
    real_objective = [s[1] for s in solution_fit]

    # print(np.correlate(objectives_cp, real_objective))
    import matplotlib.pyplot as plt
    import scipy.stats as stats

    plt.scatter(objectives_cp, real_objective)
    plt.show()
    print(annealing)


def local_search_postpro_multiobj_multimode(postpro=True):
    # file_path = files_available[0]
    files = get_data_available()
    files = [f for f in files if "j301_1.sm" in f]  # Single mode RCPSP
    # files = [f for f in files if 'j1010_5.mm' in f]  # Multi mode RCPSP
    file_path = files[0]
    rcpsp_model: MultiModeRCPSPModel = parse_file(file_path)
    # rcpsp_model.set_fixed_modes([1 for i in range(rcpsp_model.n_jobs)])
    dummy = rcpsp_model.get_dummy_solution()
    _, mutations = get_available_mutations(rcpsp_model, dummy)
    print(mutations)
    list_mutation = [
        mutate[0].build(rcpsp_model, dummy, **mutate[1])
        for mutate in mutations
        if mutate[0] == PermutationMutationRCPSP
    ]
    mixed_mutation = BasicPortfolioMutation(
        list_mutation, np.ones((len(list_mutation)))
    )
    res = RestartHandlerLimit(
        500, cur_solution=dummy, cur_objective=rcpsp_model.evaluate(dummy)
    )
    objectives = ["makespan", "mean_resource_reserve"]
    objective_weights = [-1, 5]
    if postpro:
        params_objective_function = ParamsObjectiveFunction(
            objective_handling=ObjectiveHandling.AGGREGATE,
            objectives=objectives,
            weights=objective_weights,
            sense_function=ModeOptim.MAXIMIZATION,
        )
        sa = SimulatedAnnealing(
            evaluator=rcpsp_model,
            mutator=mixed_mutation,
            restart_handler=res,
            temperature_handler=TemperatureSchedulingFactor(
                temperature=1, restart_handler=res, coefficient=0.9999
            ),
            mode_mutation=ModeMutation.MUTATE,
            params_objective_function=params_objective_function,
            store_solution=True,
            nb_solutions=10000,
        )
        result_ls = sa.solve(dummy, nb_iteration_max=2000, pickle_result=False)
    else:
        params_objective_function = ParamsObjectiveFunction(
            objective_handling=ObjectiveHandling.MULTI_OBJ,
            objectives=objectives,
            weights=objective_weights,
            sense_function=ModeOptim.MAXIMIZATION,
        )
        sa = HillClimberPareto(
            evaluator=rcpsp_model,
            mutator=mixed_mutation,
            restart_handler=res,
            params_objective_function=params_objective_function,
            mode_mutation=ModeMutation.MUTATE,
            store_solution=True,
            nb_solutions=10000,
        )
        result_ls = sa.solve(
            dummy,
            nb_iteration_max=5000,
            pickle_result=False,
            update_iteration_pareto=10000,
        )
    result_ls.list_solution_fits = [
        l for l in result_ls.list_solution_fits if l[0].rcpsp_schedule_feasible
    ]
    pareto_store = result_storage_to_pareto_front(
        result_storage=result_ls, problem=rcpsp_model
    )
    print("Nb Pareto : ", pareto_store.len_pareto_front())
    worst, average, many_random_instance = create_models(
        base_rcpsp_model=rcpsp_model,
        range_around_mean_resource=5,
        range_around_mean_duration=3,
        do_uncertain_resource=True,
        do_uncertain_duration=False,
        nb_sampled_scenario=30,
    )
    solutions_pareto = [l[0] for l in pareto_store.paretos]
    all_results = []
    results = np.zeros((len(solutions_pareto), len(many_random_instance), 3))
    import time

    from discrete_optimization.rcpsp.solver.rcpsp_pile import Executor

    executor = Executor(rcpsp_model=rcpsp_model)
    for index_instance in range(len(many_random_instance)):
        print("Evaluating in instance #", index_instance)
        instance = many_random_instance[index_instance]
        executor.update(instance)
        for index_pareto in range(len(solutions_pareto)):
            # t = time.time()
            # sol_ = RCPSPSolution(problem=instance,
            #                      rcpsp_permutation=solutions_pareto[index_pareto].rcpsp_permutation,
            #                      rcpsp_modes=solutions_pareto[index_pareto].rcpsp_modes)
            # fit = instance.evaluate(sol_)
            # t_end = time.time()
            # print(t_end-t, " seconds for classic")
            modes_dict = {1: 1}
            modes_dict[instance.n_jobs + 2] = 1
            for j in range(len(solutions_pareto[index_pareto].rcpsp_modes)):
                modes_dict[j + 2] = solutions_pareto[index_pareto].rcpsp_modes[j]
            t = time.time()
            sol_, fit = executor.compute_schedule_from_priority_list(
                permutation_jobs=[1]
                + [k + 2 for k in solutions_pareto[index_pareto].rcpsp_permutation]
                + [rcpsp_model.n_jobs + 2],
                modes_dict=modes_dict,
            )
            t_end = time.time()
            print(t_end - t, " seconds for exec")
            results[index_pareto, index_instance, 0] = (
                1 if sol_.rcpsp_schedule_feasible else 0
            )
            print(
                "schedule #",
                index_pareto,
                " succeed : ",
                results[index_pareto, index_instance, 0],
            )
            results[index_pareto, index_instance, 1] = fit["makespan"]
            results[index_pareto, index_instance, 2] = fit["mean_resource_reserve"]
    feasible = np.sum(results[:, :, 0], axis=1)
    mean_makespan = np.mean(results[:, :, 1], axis=1)
    max_makespan = np.max(results[:, :, 1], axis=1)

    mean_resource_reserve = np.array(
        [l[1].vector_fitness[1] for l in pareto_store.paretos]
    )

    print(feasible, feasible.shape)
    print(mean_makespan, mean_makespan.shape)
    fig, ax = plt.subplots(1)
    ax.scatter(x=mean_resource_reserve, y=feasible)
    ax.set_title("Nb feasible schedule function of predicted mean resource reserve")
    ax.set_xlabel("mean resource reserve")
    ax.set_ylabel("# Feasible")

    fig, ax = plt.subplots(1)
    ax.scatter(x=mean_resource_reserve, y=mean_makespan)
    ax.set_title("Mean makespan function of predicted mean resource reserve")
    ax.set_xlabel("mean resource reserve")
    ax.set_ylabel("mean makespan : ")

    fig, ax = plt.subplots(1)
    ax.scatter(x=mean_resource_reserve, y=max_makespan)
    ax.set_title("Max makespan function of predicted mean resource reserve")
    ax.set_xlabel("mean resource reserve")
    ax.set_ylabel("max makespan : ")
    plt.show()
    # plot_storage_2d(result_storage=pareto_store, name_axis=objectives, ax=ax)
    # plot_pareto_2d(pareto_front=pareto_store, name_axis=objectives, ax=ax)
    # extreme_points = pareto_store.compute_extreme_points()
    # plot_ressource_view(rcpsp_model=rcpsp_model,
    #                     rcpsp_sol=extreme_points[0][0],
    #                     title_figure="best makespan")
    # plot_resource_individual_gantt(rcpsp_model=rcpsp_model,
    #                                rcpsp_sol=extreme_points[0][0],
    #                                title_figure="best makespan")
    # plot_ressource_view(rcpsp_model=rcpsp_model,
    #                     rcpsp_sol=extreme_points[1][0],
    #                     title_figure="best availability")
    # plot_resource_individual_gantt(rcpsp_model=rcpsp_model,
    #                                rcpsp_sol=extreme_points[1][0],
    #                                title_figure="best availability")
    # plt.show()


def solve_model(model, postpro=True, nb_iteration=500):
    dummy = model.get_dummy_solution()
    _, mutations = get_available_mutations(model, dummy)
    list_mutation = [
        mutate[0].build(model, dummy, **mutate[1])
        for mutate in mutations
        if mutate[0] == PermutationMutationRCPSP
    ]
    mixed_mutation = BasicPortfolioMutation(
        list_mutation, np.ones((len(list_mutation)))
    )
    res = RestartHandlerLimit(
        500, cur_solution=dummy, cur_objective=model.evaluate(dummy)
    )
    objectives = ["makespan"]
    objective_weights = [-1]
    if postpro:
        params_objective_function = ParamsObjectiveFunction(
            objective_handling=ObjectiveHandling.AGGREGATE,
            objectives=objectives,
            weights=objective_weights,
            sense_function=ModeOptim.MAXIMIZATION,
        )
        sa = SimulatedAnnealing(
            evaluator=model,
            mutator=mixed_mutation,
            restart_handler=res,
            temperature_handler=TemperatureSchedulingFactor(
                temperature=2.0, restart_handler=res, coefficient=0.9999
            ),
            mode_mutation=ModeMutation.MUTATE,
            params_objective_function=params_objective_function,
            store_solution=True,
            nb_solutions=10000,
        )
        result_ls = sa.solve(dummy, nb_iteration_max=nb_iteration, pickle_result=False)
    else:
        params_objective_function = ParamsObjectiveFunction(
            objective_handling=ObjectiveHandling.MULTI_OBJ,
            objectives=objectives,
            weights=objective_weights,
            sense_function=ModeOptim.MAXIMIZATION,
        )
        sa = HillClimberPareto(
            evaluator=model,
            mutator=mixed_mutation,
            restart_handler=res,
            params_objective_function=params_objective_function,
            mode_mutation=ModeMutation.MUTATE,
            store_solution=True,
            nb_solutions=10000,
        )
        result_ls = sa.solve(
            dummy,
            nb_iteration_max=nb_iteration,
            pickle_result=False,
            update_iteration_pareto=10000,
        )
    return result_ls


from discrete_optimization.generic_tools.robustness.robustness_tool import (
    RobustnessTool,
)

# def evaluate():
#     modes_dict = {1: 1}
#     modes_dict[instance.n_jobs + 2] = 1
#     for j in range(len(solutions[index_pareto].rcpsp_modes)):
#         modes_dict[j + 2] = solutions[index_pareto].rcpsp_modes[j]
#     t = time.time()
#     result_store = executor.compute_schedule_from_priority_list(permutation_jobs=
#                                                                 [1]
#                                                                 +
#                                                                 [k + 2
#                                                                  for k in
#                                                                  solutions[index_pareto].rcpsp_permutation]
#                                                                 + [rcpsp_model.n_jobs + 2],
#                                                                 modes_dict=modes_dict)
#     t_end = time.time()
#     print(t_end - t, " seconds for exec")
#     sol_ = result_store.get_best_solution()
#     fit = instance.evaluate(sol_)


def local_search_aggregated(
    postpro=True, nb_iteration=100, merge_aposteriori=True, merge_apriori=True
):
    # file_path = files_available[0]
    files = get_data_available()
    # files = [f for f in files if 'j1010_1.mm' in f]  # Single mode RCPSP
    # files = [f for f in files if 'j1010_5.mm' in f]  # Multi mode RCPSP
    files = [f for f in files if "j601_9.sm" in f]  # Single mode RCPS
    file_path = files[0]
    rcpsp_model: RCPSPModel = parse_file(file_path)
    if isinstance(rcpsp_model, MultiModeRCPSPModel):
        rcpsp_model.set_fixed_modes([1 for i in range(rcpsp_model.n_jobs)])
    worst, average, many_random_instance = create_models(
        base_rcpsp_model=rcpsp_model,
        range_around_mean_resource=1,
        range_around_mean_duration=2,
        do_uncertain_resource=False,
        do_uncertain_duration=True,
        nb_sampled_scenario=1000,
    )
    import random

    len_random_instance = len(many_random_instance)
    random.shuffle(many_random_instance)
    proportion_train = 0.8
    len_train = int(proportion_train * len_random_instance)
    train_data = many_random_instance[:len_train]
    test_data = many_random_instance[len_train:]
    robust = RobustnessTool(
        base_instance=rcpsp_model,
        all_instances=many_random_instance,
        train_instance=train_data,
        test_instance=test_data,
    )
    models = robust.get_models(apriori=True, aposteriori=True)
    from functools import partial

    solve_function = partial(solve_model, postpro=postpro, nb_iteration=nb_iteration)
    results = robust.solve_and_retrieve(solve_models_function=solve_function)
    robust.plot(results, image_tag="testing-robust")


if __name__ == "__main__":
    run_cp_multiscenario()
    # local_search_aggregated(postpro=True,
    #                         nb_iteration=2000,
    #                         merge_apriori=True,
    #                         merge_aposteriori=True)
