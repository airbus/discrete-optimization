#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import random
import time
from collections import defaultdict
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from discrete_optimization.generic_tools.cp_tools import CpSolverName, ParametersCp
from discrete_optimization.generic_tools.do_problem import (
    BaseMethodAggregating,
    MethodAggregating,
    ModeOptim,
    ObjectiveHandling,
    ParamsObjectiveFunction,
)
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
    PermutationMutationRcpsp,
    get_available_mutations,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    result_storage_to_pareto_front,
)
from discrete_optimization.generic_tools.robustness.robustness_tool import (
    RobustnessTool,
)
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp.problem_robust import (
    AggregRcpspProblem,
    MethodBaseRobustification,
    MethodRobustification,
    UncertainRcpspProblem,
    create_poisson_laws_duration,
    create_poisson_laws_resource,
)
from discrete_optimization.rcpsp.solution import RcpspSolution
from discrete_optimization.rcpsp.solvers.cp_mzn_multiscenario import (
    CpMultiscenarioRcpspSolver,
)
from discrete_optimization.rcpsp.solvers.pile import Executor

logging.basicConfig(level=logging.DEBUG)


def tree():
    return defaultdict(tree)


def create_models(
    base_rcpsp_problem: RcpspProblem,
    range_around_mean_resource: int = 1,
    range_around_mean_duration: int = 3,
    do_uncertain_resource: bool = True,
    do_uncertain_duration: bool = True,
    nb_sampled_scenario: int = 50,
):
    poisson_laws = tree()
    if do_uncertain_duration:
        poisson_laws_duration = create_poisson_laws_duration(
            base_rcpsp_problem, range_around_mean=range_around_mean_duration
        )
        for job in poisson_laws_duration:
            for mode in poisson_laws_duration[job]:
                for res in poisson_laws_duration[job][mode]:
                    poisson_laws[job][mode][res] = poisson_laws_duration[job][mode][res]
    if do_uncertain_resource:
        poisson_laws_resource = create_poisson_laws_resource(
            base_rcpsp_problem, range_around_mean=range_around_mean_resource
        )
        for job in poisson_laws_resource:
            for mode in poisson_laws_resource[job]:
                for res in poisson_laws_resource[job][mode]:
                    poisson_laws[job][mode][res] = poisson_laws_resource[job][mode][res]

    uncertain = UncertainRcpspProblem(base_rcpsp_problem, poisson_laws=poisson_laws)
    worst = uncertain.create_rcpsp_problem(
        MethodRobustification(MethodBaseRobustification.WORST_CASE, percentile=0)
    )
    average = uncertain.create_rcpsp_problem(
        MethodRobustification(MethodBaseRobustification.AVERAGE, percentile=0)
    )
    many_random_instance = [
        uncertain.create_rcpsp_problem(
            method_robustification=MethodRobustification(
                MethodBaseRobustification.SAMPLE
            )
        )
        for i in range(nb_sampled_scenario)
    ]
    return worst, average, many_random_instance


def run_cp_multiscenario():
    files = get_data_available()
    files = [f for f in files if "j301_1.sm" in f]  # Single mode RCPSP
    print(files)
    file_path = files[0]
    rcpsp_problem = parse_file(file_path)
    poisson_laws = create_poisson_laws_duration(rcpsp_problem, range_around_mean=2)
    uncertain_model: UncertainRcpspProblem = UncertainRcpspProblem(
        base_rcpsp_problem=rcpsp_problem, poisson_laws=poisson_laws
    )
    list_rcpsp_problem = [
        uncertain_model.create_rcpsp_problem(
            MethodRobustification(
                method_base=MethodBaseRobustification.SAMPLE, percentile=0
            )
        )
        for i in range(20)
    ]

    for m in list_rcpsp_problem:
        m.update_functions()
    for model in list_rcpsp_problem:
        model.costs["mean_resource_reserve"] = True
    dummy = list_rcpsp_problem[0].get_dummy_solution()
    model_aggreg_mean = AggregRcpspProblem(
        list_problem=list_rcpsp_problem,
        method_aggregating=MethodAggregating(BaseMethodAggregating.MEAN),
    )
    _, mutations = get_available_mutations(list_rcpsp_problem[0], dummy)
    print(mutations)
    list_mutation = [
        mutate[0].build(rcpsp_problem, dummy, **mutate[1])
        for mutate in mutations
        if mutate[0] == PermutationMutationRcpsp
    ]
    mixed_mutation = BasicPortfolioMutation(
        list_mutation, np.ones((len(list_mutation)))
    )
    res = RestartHandlerLimit(500)
    simulated_annealing = SimulatedAnnealing(
        problem=model_aggreg_mean,
        mutator=mixed_mutation,
        mode_mutation=ModeMutation.MUTATE,
        restart_handler=res,
        temperature_handler=TemperatureSchedulingFactor(
            temperature=1, restart_handler=res, coefficient=0.9999
        ),
        store_solution=True,
    )
    result_sa = simulated_annealing.solve(initial_variable=dummy, nb_iteration_max=300)
    best_solution: RcpspSolution = result_sa.get_best_solution()
    permutation = best_solution.rcpsp_permutation
    annealing = []
    for model in list_rcpsp_problem:
        s = RcpspSolution(problem=model, rcpsp_permutation=permutation)
        annealing += [model.evaluate(s)["makespan"]]

    solver = CpMultiscenarioRcpspSolver(
        problem=model_aggreg_mean, cp_solver_name=CpSolverName.CHUFFED
    )
    solver.init_model(
        output_type=True, relax_ordering=False, nb_incoherence_limit=2, max_time=300
    )
    params_cp = ParametersCp.default()
    params_cp.free_search = True
    result = solver.solve(parameters_cp=params_cp, time_limit=50)
    objectives_cp = [sol.minizinc_obj for sol, fit in result]
    real_objective = [fit for sol, fit in result]

    plt.scatter(objectives_cp, real_objective)
    plt.show()
    print(annealing)


def local_search_postpro_multiobj_multimode(postpro=True):
    files = get_data_available()
    files = [f for f in files if "j301_1.sm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_problem = parse_file(file_path)
    dummy = rcpsp_problem.get_dummy_solution()
    _, mutations = get_available_mutations(rcpsp_problem, dummy)
    print(mutations)
    list_mutation = [
        mutate[0].build(rcpsp_problem, dummy, **mutate[1])
        for mutate in mutations
        if mutate[0] == PermutationMutationRcpsp
    ]
    mixed_mutation = BasicPortfolioMutation(
        list_mutation, np.ones((len(list_mutation)))
    )
    res = RestartHandlerLimit(500)
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
            problem=rcpsp_problem,
            mutator=mixed_mutation,
            restart_handler=res,
            temperature_handler=TemperatureSchedulingFactor(
                temperature=1, restart_handler=res, coefficient=0.9999
            ),
            mode_mutation=ModeMutation.MUTATE,
            params_objective_function=params_objective_function,
            store_solution=True,
        )
        result_ls = sa.solve(initial_variable=dummy, nb_iteration_max=2000)
    else:
        params_objective_function = ParamsObjectiveFunction(
            objective_handling=ObjectiveHandling.MULTI_OBJ,
            objectives=objectives,
            weights=objective_weights,
            sense_function=ModeOptim.MAXIMIZATION,
        )
        sa = HillClimberPareto(
            problem=rcpsp_problem,
            mutator=mixed_mutation,
            restart_handler=res,
            params_objective_function=params_objective_function,
            mode_mutation=ModeMutation.MUTATE,
            store_solution=True,
        )
        result_ls = sa.solve(
            initial_variable=dummy,
            nb_iteration_max=5000,
            update_iteration_pareto=10000,
        )
    result_ls = [l for l in result_ls if l[0].rcpsp_schedule_feasible]
    pareto_store = result_storage_to_pareto_front(
        result_storage=result_ls, problem=rcpsp_problem
    )
    print("Nb Pareto : ", pareto_store.len_pareto_front())
    worst, average, many_random_instance = create_models(
        base_rcpsp_problem=rcpsp_problem,
        range_around_mean_resource=5,
        range_around_mean_duration=3,
        do_uncertain_resource=True,
        do_uncertain_duration=False,
        nb_sampled_scenario=30,
    )
    solutions_pareto = [l[0] for l in pareto_store.paretos]
    all_results = []
    results = np.zeros((len(solutions_pareto), len(many_random_instance), 3))

    executor = Executor(problem=rcpsp_problem)
    for index_instance in range(len(many_random_instance)):
        print("Evaluating in instance #", index_instance)
        instance = many_random_instance[index_instance]
        executor.update(instance)
        for index_pareto in range(len(solutions_pareto)):
            modes_dict = {1: 1}
            modes_dict[instance.n_jobs + 2] = 1
            for j in range(len(solutions_pareto[index_pareto].rcpsp_modes)):
                modes_dict[j + 2] = solutions_pareto[index_pareto].rcpsp_modes[j]
            t = time.time()
            sol_, fit = executor.compute_schedule_from_priority_list(
                permutation_jobs=[1]
                + [k + 2 for k in solutions_pareto[index_pareto].rcpsp_permutation]
                + [rcpsp_problem.n_jobs + 2],
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


def solve_model(model, postpro=True, nb_iteration=500):
    dummy = model.get_dummy_solution()
    _, mutations = get_available_mutations(model, dummy)
    list_mutation = [
        mutate[0].build(model, dummy, **mutate[1])
        for mutate in mutations
        if mutate[0] == PermutationMutationRcpsp
    ]
    mixed_mutation = BasicPortfolioMutation(
        list_mutation, np.ones((len(list_mutation)))
    )
    res = RestartHandlerLimit(500)
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
            problem=model,
            mutator=mixed_mutation,
            restart_handler=res,
            temperature_handler=TemperatureSchedulingFactor(
                temperature=2.0, restart_handler=res, coefficient=0.9999
            ),
            mode_mutation=ModeMutation.MUTATE,
            params_objective_function=params_objective_function,
            store_solution=True,
        )
        result_ls = sa.solve(initial_variable=dummy, nb_iteration_max=nb_iteration)
    else:
        params_objective_function = ParamsObjectiveFunction(
            objective_handling=ObjectiveHandling.MULTI_OBJ,
            objectives=objectives,
            weights=objective_weights,
            sense_function=ModeOptim.MAXIMIZATION,
        )
        sa = HillClimberPareto(
            problem=model,
            mutator=mixed_mutation,
            restart_handler=res,
            params_objective_function=params_objective_function,
            mode_mutation=ModeMutation.MUTATE,
            store_solution=True,
        )
        result_ls = sa.solve(
            initial_variable=dummy,
            nb_iteration_max=nb_iteration,
            update_iteration_pareto=10000,
        )
    return result_ls


def local_search_aggregated(
    postpro=True, nb_iteration=100, merge_aposteriori=True, merge_apriori=True
):
    files = get_data_available()
    files = [f for f in files if "j601_9.sm" in f]  # Single mode RCPS
    file_path = files[0]
    rcpsp_problem: RcpspProblem = parse_file(file_path)
    if rcpsp_problem.is_rcpsp_multimode():
        rcpsp_problem.set_fixed_modes([1 for i in range(rcpsp_problem.n_jobs)])
    worst, average, many_random_instance = create_models(
        base_rcpsp_problem=rcpsp_problem,
        range_around_mean_resource=1,
        range_around_mean_duration=2,
        do_uncertain_resource=False,
        do_uncertain_duration=True,
        nb_sampled_scenario=1000,
    )

    len_random_instance = len(many_random_instance)
    random.shuffle(many_random_instance)
    proportion_train = 0.8
    len_train = int(proportion_train * len_random_instance)
    train_data = many_random_instance[:len_train]
    test_data = many_random_instance[len_train:]
    robust = RobustnessTool(
        base_instance=rcpsp_problem,
        all_instances=many_random_instance,
        train_instance=train_data,
        test_instance=test_data,
    )
    models = robust.get_models(apriori=True, aposteriori=True)

    solve_function = partial(solve_model, postpro=postpro, nb_iteration=nb_iteration)
    results = robust.solve_and_retrieve(solve_models_function=solve_function)
    robust.plot(results, image_tag="testing-robust")


if __name__ == "__main__":
    run_cp_multiscenario()
