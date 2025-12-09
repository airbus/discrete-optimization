#  Copyright (c) 2022-2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import random
import time
from collections import defaultdict
from functools import partial
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
from discrete_optimization.generic_tools.mutations.mutation_catalog import (
    RcpspMutation,
)
from discrete_optimization.generic_tools.mutations.mutation_portfolio import (
    create_mutations_portfolio_from_problem,
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

logging.basicConfig(level=logging.INFO)


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
    mixed_mutation = create_mutations_portfolio_from_problem(
        problem=rcpsp_problem, selected_mutations={RcpspMutation}
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
    mixed_mutation = create_mutations_portfolio_from_problem(
        problem=rcpsp_problem, selected_mutations={RcpspMutation}
    )
    res = RestartHandlerLimit(500)
    objectives = ["makespan", "mean_resource_reserve"]
    objective_weights = [-1, 5]
    rcpsp_problem.compute_mean_resource = True
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
    from discrete_optimization.generic_tools.result_storage.result_storage import (
        ResultStorage,
    )

    result_ls = ResultStorage(
        mode_optim=result_ls.mode_optim,
        list_solution_fits=[l for l in result_ls if l[0].rcpsp_schedule_feasible],
    )
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
    rcpsp_problem.compute_mean_resource = True
    for index_instance in range(len(many_random_instance)):
        print("Evaluating in instance #", index_instance)
        instance = many_random_instance[index_instance]
        executor.update(instance)
        instance.compute_mean_resource = True
        for index_pareto in range(len(solutions_pareto)):
            modes_dict = {1: 1}
            modes_dict[instance.n_jobs_non_dummy + 2] = 1
            for j in range(len(solutions_pareto[index_pareto].rcpsp_modes)):
                modes_dict[j + 2] = solutions_pareto[index_pareto].rcpsp_modes[j]
            t = time.time()
            res = executor.compute_schedule_from_priority_list(
                permutation_jobs=[1]
                + [k + 2 for k in solutions_pareto[index_pareto].rcpsp_permutation]
                + [rcpsp_problem.n_jobs_non_dummy + 2],
                modes_dict=modes_dict,
            )
            sol_, fit = res.get_best_solution_fit()
            metrics = instance.evaluate(sol_)
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
            results[index_pareto, index_instance, 1] = metrics["makespan"]
            results[index_pareto, index_instance, 2] = metrics["mean_resource_reserve"]
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


def solve_model(
    model: Union[AggregRcpspProblem, RcpspProblem], postpro=True, nb_iteration=500
):
    if isinstance(model, AggregRcpspProblem):
        dummy = model.get_dummy_solution()
        mixed_mutation = create_mutations_portfolio_from_problem(
            problem=model, selected_mutations={RcpspMutation}
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
    else:
        from discrete_optimization.generic_tools.cp_tools import ParametersCp
        from discrete_optimization.rcpsp.solvers.cpsat import CpSatRcpspSolver

        solver = CpSatRcpspSolver(problem=model)
        solver.init_model()
        p = ParametersCp.default_cpsat()
        p.nb_process = 16
        res = solver.solve(
            parameters_cp=p,
            time_limit=5,
            ortools_cpsat_solver_kwargs={"log_search_progress": True},
        )
        return res


def local_search_multiscenario():
    # In case of fail, run discrete_optimization.datasets script
    files = get_data_available()
    files = [f for f in files if "j301_1.sm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_problem = parse_file(file_path)
    # Create some scenarios around the initial problem.
    # Worst is the worst case scenario
    # Average is some representative average value
    # Many random instance: is N sample from the uncertainty distribution.
    worst, average, many_random_instance = create_models(
        base_rcpsp_problem=rcpsp_problem,
        range_around_mean_resource=1,
        range_around_mean_duration=2,
        do_uncertain_resource=False,
        do_uncertain_duration=True,
        nb_sampled_scenario=50,
    )
    dummy = many_random_instance[0].get_dummy_solution()
    model_aggreg_mean = AggregRcpspProblem(
        list_problem=many_random_instance,
        method_aggregating=MethodAggregating(BaseMethodAggregating.MEAN),
    )
    _, mutations = get_available_mutations(many_random_instance[0], dummy)
    mixed_mutation = create_mutations_portfolio_from_problem(
        problem=rcpsp_problem, selected_mutations={RcpspMutation}
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
    result_sa = simulated_annealing.solve(initial_variable=dummy, nb_iteration_max=1000)
    best_solution: RcpspSolution = result_sa.get_best_solution()
    print("Best found permutation ", best_solution.rcpsp_permutation)
    permutation = best_solution.rcpsp_permutation
    annealing_makespan = []
    for model in many_random_instance:
        s = RcpspSolution(problem=model, rcpsp_permutation=permutation)
        annealing_makespan += [model.evaluate(s)["makespan"]]
    print(annealing_makespan)


def local_search_aggregated(
    postpro=True, nb_iteration=100, merge_aposteriori=True, merge_apriori=True
):
    files = get_data_available()
    files = [f for f in files if "j1201_9.sm" in f]
    file_path = files[0]
    rcpsp_problem: RcpspProblem = parse_file(file_path)

    if rcpsp_problem.is_rcpsp_multimode():
        rcpsp_problem.set_fixed_modes([1 for i in range(rcpsp_problem.n_jobs)])

    # 1. Create Uncertain Models
    worst, average, many_random_instance = create_models(
        base_rcpsp_problem=rcpsp_problem,
        range_around_mean_resource=1,
        range_around_mean_duration=2,
        do_uncertain_resource=False,
        do_uncertain_duration=True,
        nb_sampled_scenario=100,
    )

    # 2. Split Data
    len_random_instance = len(many_random_instance)
    random.shuffle(many_random_instance)
    proportion_train = 0.8
    len_train = int(proportion_train * len_random_instance)
    train_data = many_random_instance[:len_train]
    test_data = many_random_instance[len_train:]

    # 3. Initialize Robustness Tool
    robust = RobustnessTool(
        base_instance=rcpsp_problem,
        all_instances=many_random_instance,
        train_instance=train_data,
        test_instance=test_data,
    )

    # 4. Solve
    solve_function = partial(solve_model, postpro=postpro, nb_iteration=nb_iteration)
    results = robust.solve_and_retrieve(
        solve_models_function=solve_function,
        apriori=merge_apriori,
        aposteriori=merge_aposteriori,
    )

    # A. Statistics Table
    print("\n=== Robustness Statistics ===")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    df_stats = robust.get_statistics_df(results)
    print(df_stats)
    print("=============================\n")
    # Dist/Histogram to see the makespan over scenarios
    robust.plot(results, image_tag="experiment_J60")
    # boxplot view (clearer!)
    robust.plot_boxplots(results, image_tag="experiment_J60")

    # Look at schedules found on diff scenarios.
    # Look at the Robust method (usually post_max) vs Average method
    # if "post_max" in robust.tags:
    #     robust.visualize_scenarios(method_tag="post_max", nb_scenarios=3)

    # if "post_mean" in robust.tags:
    #    robust.visualize_scenarios(method_tag="post_mean", nb_scenarios=3)


if __name__ == "__main__":
    local_search_aggregated(nb_iteration=500)
