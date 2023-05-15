#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random

import matplotlib.pyplot as plt

from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveHandling,
    ParamsObjectiveFunction,
    build_evaluate_function_aggregated,
)
from discrete_optimization.generic_tools.ea.ga import DeapMutation
from discrete_optimization.generic_tools.ea.nsga import Nsga
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.generic_tools.result_storage.resultcomparator import (
    ResultComparator,
)
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file
from discrete_optimization.rcpsp.rcpsp_solution import RCPSPSolution
from discrete_optimization.rcpsp.robust_rcpsp import (
    MethodBaseRobustification,
    MethodRobustification,
    UncertainRCPSPModel,
    create_poisson_laws,
)


def run_single_mode_moo_benchmark():

    # Problem initialisation
    files = get_data_available()
    files = [f for f in files if "j301_1.sm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_model = parse_file(file_path)
    rcpsp_model.costs["mean_resource_reserve"] = True

    # Objective settings
    objectives = ["makespan", "mean_resource_reserve"]
    objective_weights = [-1, +1]

    # Algo 1: NSGA
    mutation = DeapMutation.MUT_SHUFFLE_INDEXES
    ga_solver = Nsga(
        rcpsp_model,
        encoding="rcpsp_permutation",
        objectives=objectives,
        objective_weights=objective_weights,
        mutation=mutation,
    )
    ga_solver._max_evals = 2000
    result_storage_1 = ga_solver.solve()

    # Algo 2: NSGA - few evaluations
    mutation = DeapMutation.MUT_SHUFFLE_INDEXES
    ga_solver = Nsga(
        rcpsp_model,
        encoding="rcpsp_permutation",
        objectives=objectives,
        objective_weights=objective_weights,
        mutation=mutation,
    )
    ga_solver._max_evals = 200
    result_storage_2 = ga_solver.solve()

    # Setting up the ResultComparator
    list_result_storage = [result_storage_1, result_storage_2]
    for rs in list_result_storage:
        rs.remove_duplicate_solutions(var_name="standardised_permutation")
    result_storage_names = ["nsga_many_evals", "nsga_few_evals"]
    objectives_str = objectives
    test_problems = None
    result_comparator = ResultComparator(
        list_result_storage=list_result_storage,
        result_storage_names=result_storage_names,
        objectives_str=objectives_str,
        objective_weights=objective_weights,
        test_problems=test_problems,
    )

    # Get best by individual objective:
    print(
        "Best makespan : ",
        result_comparator.get_best_by_objective_by_result_storage("makespan"),
    )
    print(
        "Best resource reserve",
        result_comparator.get_best_by_objective_by_result_storage(
            "mean_resource_reserve"
        ),
    )

    # Plot the same information
    result_comparator.plot_all_best_by_objective("makespan")
    plt.show()

    result_comparator.plot_all_best_by_objective("mean_resource_reserve")
    plt.show()

    result_comparator.plot_all_2d_paretos_single_plot(objectives_str=objectives)
    plt.show()

    result_comparator.plot_all_2d_paretos_subplots()
    plt.show()

    # Generate and plot super Pareto front, the one to rule them all
    result_comparator.plot_super_pareto()
    plt.show()


def run_single_mode_robustness_benchmark():

    # Problem initialisation
    files = get_data_available()
    files = [f for f in files if "j301_1.sm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_model = parse_file(file_path)
    rcpsp_model.costs["mean_resource_reserve"] = True
    objectives = ["makespan", "mean_resource_reserve"]
    objective_weights = [-1, 1]

    # 3 random solutions:
    sol_perm_1 = [i for i in range(rcpsp_model.n_jobs_non_dummy)]
    sol_perm_2 = [
        rcpsp_model.n_jobs_non_dummy - 1 - i
        for i in range(rcpsp_model.n_jobs_non_dummy)
    ]
    sol_perm_3 = [i for i in range(rcpsp_model.n_jobs_non_dummy)]
    random.shuffle(sol_perm_3)

    print("sol_perm_1:", sol_perm_1)
    print("sol_perm_2:", sol_perm_2)
    print("sol_perm_3:", sol_perm_3)

    sol_1 = RCPSPSolution(problem=rcpsp_model, rcpsp_permutation=sol_perm_1)
    sol_2 = RCPSPSolution(problem=rcpsp_model, rcpsp_permutation=sol_perm_2)
    sol_3 = RCPSPSolution(problem=rcpsp_model, rcpsp_permutation=sol_perm_3)

    print("sol_1:", sol_1)
    print("sol_2:", sol_2)
    print("sol_3:", sol_3)

    # Setting up 1 result storage per solution
    params_objective_function = ParamsObjectiveFunction(
        objective_handling=ObjectiveHandling.MULTI_OBJ,
        objectives=objectives,
        weights=objective_weights,
        sense_function=ModeOptim.MAXIMIZATION,
    )
    evaluate_sol, _ = build_evaluate_function_aggregated(
        problem=rcpsp_model, params_objective_function=params_objective_function
    )
    sols = []
    s_pure_int = [i for i in sol_1.rcpsp_permutation]
    kwargs = {"rcpsp_permutation": s_pure_int, "problem": rcpsp_model}
    problem_sol = rcpsp_model.get_solution_type()(**kwargs)
    fits = evaluate_sol(problem_sol)
    sols.append((problem_sol, fits))
    rs_1 = ResultStorage(list_solution_fits=sols, best_solution=None)

    sols = []
    s_pure_int = [i for i in sol_2.rcpsp_permutation]
    kwargs = {"rcpsp_permutation": s_pure_int, "problem": rcpsp_model}
    problem_sol = rcpsp_model.get_solution_type()(**kwargs)
    fits = evaluate_sol(problem_sol)
    sols.append((problem_sol, fits))
    rs_2 = ResultStorage(list_solution_fits=sols, best_solution=None)

    sols = []
    s_pure_int = [i for i in sol_3.rcpsp_permutation]
    kwargs = {"rcpsp_permutation": s_pure_int, "problem": rcpsp_model}
    problem_sol = rcpsp_model.get_solution_type()(**kwargs)
    fits = evaluate_sol(problem_sol)
    sols.append((problem_sol, fits))
    rs_3 = ResultStorage(list_solution_fits=sols, best_solution=None)

    # Setting up test scenarios
    poisson_laws = create_poisson_laws(
        rcpsp_model,
        range_around_mean_duration=2,
        range_around_mean_resource=1,
        do_uncertain_duration=True,
        do_uncertain_resource=False,
    )
    uncertain_model: UncertainRCPSPModel = UncertainRCPSPModel(
        base_rcpsp_model=rcpsp_model, poisson_laws=poisson_laws
    )

    many_random_instance = [
        uncertain_model.create_rcpsp_model(
            method_robustification=MethodRobustification(
                MethodBaseRobustification.SAMPLE
            )
        )
        for i in range(300)
    ]
    for model in many_random_instance:
        model.costs["mean_resource_reserve"] = True
    len_random_instance = len(many_random_instance)
    random.shuffle(many_random_instance)
    proportion_train = 0.8
    len_train = int(proportion_train * len_random_instance)
    train_data = many_random_instance[:len_train]
    test_data = many_random_instance[len_train:]

    # Setting up the ResultComparator
    list_result_storage = [rs_1, rs_2, rs_3]
    result_storage_names = ["sol_1", "sol_2", "sol_3"]
    objectives_str = objectives
    test_problems = test_data
    result_comparator = ResultComparator(
        list_result_storage=list_result_storage,
        result_storage_names=result_storage_names,
        objectives_str=objectives_str,
        objective_weights=objective_weights,
        test_problems=test_problems,
    )
    result_comparator.plot_distribution_for_objective(objective_str=objectives[0])
    result_comparator.plot_distribution_for_objective(objective_str=objectives[1])
    plt.show()


if __name__ == "__main__":
    run_single_mode_moo_benchmark()
    run_single_mode_robustness_benchmark()
