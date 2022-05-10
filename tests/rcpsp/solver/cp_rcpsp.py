from discrete_optimization.generic_tools.cp_tools import CPSolverName, ParametersCP
from discrete_optimization.generic_tools.result_storage.result_storage import (
    plot_pareto_2d,
    plot_storage_2d,
    result_storage_to_pareto_front,
)
from discrete_optimization.rcpsp.rcpsp_data_generator import (
    generate_rcpsp_with_helper_tasks_data,
)
from discrete_optimization.rcpsp.rcpsp_model import (
    MethodBaseRobustification,
    MethodRobustification,
    PartialSolution,
    RCPSPModel,
    RCPSPSolution,
    UncertainRCPSPModel,
    create_poisson_laws_duration,
)
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file
from discrete_optimization.rcpsp.rcpsp_utils import (
    kendall_tau_similarity,
    plot_task_gantt,
)
from discrete_optimization.rcpsp.solver.cp_solvers import CP_MRCPSP_MZN, CP_RCPSP_MZN


def single_mode_rcpsp_cp():
    files_available = get_data_available()
    file = [f for f in files_available if "j1201_5.sm" in f][0]
    rcpsp_problem = parse_file(file)
    solver = CP_RCPSP_MZN(rcpsp_problem, cp_solver_name=CPSolverName.CHUFFED)
    solver.init_model(output_type=True)
    result_storage = solver.solve(limit_time_s=30, verbose=True)
    solution, fit = result_storage.get_best_solution_fit()
    solution_rebuilt = RCPSPSolution(
        problem=rcpsp_problem, rcpsp_permutation=solution.rcpsp_permutation
    )
    fit_2 = rcpsp_problem.evaluate(solution_rebuilt)
    print(fit, fit_2)
    print("Satisfy : ", rcpsp_problem.satisfy(solution))
    rcpsp_problem.plot_ressource_view(solution)
    plot_task_gantt(rcpsp_problem, solution)
    import matplotlib.pyplot as plt

    plt.show()


def single_mode_rcpsp_cp_intermediate_solution():
    files_available = get_data_available()
    file = [f for f in files_available if "j1201_1.sm" in f][0]
    rcpsp_problem = parse_file(file)
    solver = CP_RCPSP_MZN(rcpsp_problem, cp_solver_name=CPSolverName.CHUFFED)
    solver.init_model(output_type=True)
    parameters_cp = ParametersCP.default()
    parameters_cp.TimeLimit = 200
    result_storage = solver.solve(
        parameters_cp=parameters_cp, output_type=True, verbose=True
    )
    best = result_storage.get_best_solution_fit()
    pareto_store = result_storage_to_pareto_front(
        result_storage=result_storage, problem=rcpsp_problem
    )
    # filename = 'j301_1.sm_postpro_dummy_cp.pkl'
    # import os, pickle
    # if os.path.exists(filename):
    #     os.remove(filename)
    # thefile = open(filename, 'ab')
    # pickle.dump(pareto_store, thefile)
    # thefile.close()
    print(len(result_storage.list_solution_fits))
    print(pareto_store.len_pareto_front())
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1)
    plot_storage_2d(
        result_storage=pareto_store,
        name_axis=["makespan", "mean_resource_reserve"],
        ax=ax,
    )
    plot_pareto_2d(
        pareto_front=pareto_store,
        name_axis=["makespan", "mean_resource_reserve"],
        ax=ax,
    )
    plt.show()


def create_models(base_rcpsp_model: RCPSPModel, range_around_mean: int = 3):
    poisson_laws = create_poisson_laws_duration(
        base_rcpsp_model, range_around_mean=range_around_mean
    )
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
        for i in range(50)
    ]
    # many_random_instance = []
    many_random_instance += [
        uncertain.create_rcpsp_model(
            method_robustification=MethodRobustification(
                MethodBaseRobustification.PERCENTILE, percentile=j
            )
        )
        for j in range(50, 100)
    ]
    return worst, average, many_random_instance


def run_cp_robust():
    files = get_data_available()
    files = [f for f in files if "j1201_1.sm" in f]  #
    file_path = files[0]
    rcpsp_model: RCPSPModel = parse_file(file_path)
    worst, average, many_random_instance = create_models(
        rcpsp_model, range_around_mean=5
    )
    solver_worst = CP_RCPSP_MZN(rcpsp_model=worst)
    solver_average = CP_RCPSP_MZN(rcpsp_model=average)
    solver_original = CP_RCPSP_MZN(rcpsp_model=rcpsp_model)
    sol_original, fit_original = solver_original.solve(limit_time_s=20)
    print(fit_original, "fitness found on original case")
    sol_worst, fit_worst = solver_worst.solve(limit_time_s=20)
    print(fit_worst, "fitness found on worst case")
    sol_average, fit_average = solver_average.solve(limit_time_s=20)
    print(fit_average, "fitness found on average case")
    permutation_worst = sol_worst.rcpsp_permutation
    permutation_original = sol_original.rcpsp_permutation
    permutation_average = sol_average.rcpsp_permutation
    fits_worst = []
    fits_original = []
    fits_average = []
    for instance in many_random_instance:
        sol_ = RCPSPSolution(problem=instance, rcpsp_permutation=permutation_original)
        fit = instance.evaluate(sol_)
        fits_original += [fit]
        sol_ = RCPSPSolution(problem=instance, rcpsp_permutation=permutation_average)
        fit = instance.evaluate(sol_)
        fits_average += [fit]
        sol_ = RCPSPSolution(problem=instance, rcpsp_permutation=permutation_worst)
        fit = instance.evaluate(sol_)
        fits_worst += [fit]

    sol_ = RCPSPSolution(problem=rcpsp_model, rcpsp_permutation=permutation_worst)
    fit = rcpsp_model.evaluate(sol_)
    print("Fit on original problem worst :", fit)

    sol_ = RCPSPSolution(problem=rcpsp_model, rcpsp_permutation=permutation_original)
    fit = rcpsp_model.evaluate(sol_)
    print("Fit on original problem origin :", fit)

    sol_ = RCPSPSolution(problem=rcpsp_model, rcpsp_permutation=permutation_average)
    fit = rcpsp_model.evaluate(sol_)
    print("Fit on original problem average :", fit)
    import numpy as np
    import scipy.stats
    import seaborn as sns

    makespans_worst = np.array([f["makespan"] for f in fits_worst])
    makespans_average = np.array([f["makespan"] for f in fits_average])
    makespans_origin = np.array([f["makespan"] for f in fits_original])
    print(
        "Stats from robust worst case version : ", scipy.stats.describe(makespans_worst)
    )
    print(min(makespans_worst), max(makespans_worst))
    print("Stats from original model", scipy.stats.describe(makespans_origin))
    print(min(makespans_origin), max(makespans_origin))
    print("Stats from average model", scipy.stats.describe(makespans_average))
    print(min(makespans_average), max(makespans_average))
    import matplotlib.pyplot as plt

    sns.distplot(
        makespans_worst,
        rug=True,
        bins=max(1, len(many_random_instance) // 10),
        label="Robust worst case",
    )
    sns.distplot(
        makespans_average,
        rug=True,
        bins=max(1, len(many_random_instance) // 10),
        label="Robust average case",
    )
    sns.distplot(
        makespans_origin,
        rug=True,
        bins=max(1, len(many_random_instance) // 10),
        label="Original",
    )
    plt.legend()
    plt.show()

    ktd = kendall_tau_similarity((sol_average, sol_worst))
    print("Kendall-Tau distance between permutation of the 2 schedules: ", ktd)


def compare_integer_and_bool_models():
    import os
    import time

    from discrete_optimization.rcpsp.solver.cp_solvers import (
        CP_MRCPSP_MZN,
        CP_MRCPSP_MZN_NOBOOL,
        CP_RCPSP_MZN,
    )

    times_dict = {}
    results_dict = {}
    files_available = get_data_available()
    # files_to_run = [f for f in files_available if 'j1010_10_2.mm' in f]
    files_to_run = files_available
    for f in files_to_run:
        if "mm" not in f:
            continue
        rcpsp_problem = parse_file(f)
        times_dict[os.path.basename(f)] = {}
        results_dict[os.path.basename(f)] = {}
        for solver_name in [CP_MRCPSP_MZN, CP_MRCPSP_MZN_NOBOOL]:
            solver = solver_name(rcpsp_problem)
            solver.init_model()
            t = time.time()
            result_storage = solver.solve(limit_time_s=100, verbose=True)
            t_end = time.time()
            solution = result_storage.get_best_solution()
            print(solution)
            makespan = rcpsp_problem.evaluate(solution)["makespan"]
            results_dict[os.path.basename(f)][solver.__class__.__name__] = (
                t_end - t,
                makespan,
            )
            print(results_dict[os.path.basename(f)])
    print(results_dict)


def multi_mode_rcpsp_cp_intermediate_solution():
    files_available = get_data_available()
    file = [f for f in files_available if "j1010_1.mm" in f][0]
    rcpsp_problem = parse_file(file)
    solver = CP_MRCPSP_MZN(rcpsp_problem, cp_solver_name=CPSolverName.CHUFFED)
    result_storage = solver.solve(parameters_cp=ParametersCP.default(), verbose=True)
    pareto_store = result_storage_to_pareto_front(
        result_storage=result_storage, problem=rcpsp_problem
    )
    print(len(result_storage.list_solution_fits))
    print(pareto_store.len_pareto_front())
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1)
    plot_storage_2d(
        result_storage=pareto_store,
        name_axis=["makespan", "mean_resource_reserve"],
        ax=ax,
    )
    plot_pareto_2d(
        pareto_front=pareto_store,
        name_axis=["makespan", "mean_resource_reserve"],
        ax=ax,
    )
    plt.show()


def single_mode_robot():
    files = get_data_available()
    files = [f for f in files if "j601_5.sm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_model = parse_file(file_path)
    # Some settings
    original_duration_multiplier = 1
    n_assisted_activities = 30
    n_assistants = 1
    probability_of_cross_helper_precedences = 0.5
    fixed_helper_duration = 2
    random_seed = 4
    # rcpsp_h = generate_rcpsp_with_helper_tasks_data(rcpsp_model,
    #                                                 original_duration_multiplier,
    #                                                 n_assisted_activities, n_assistants,
    #                                                 probability_of_cross_helper_precedences,
    #                                                 fixed_helper_duration=fixed_helper_duration,
    #                                                 random_seed=random_seed)
    # graph = rcpsp_h.compute_graph()
    # cycles = graph.check_loop()
    # print(cycles)
    solver = CP_RCPSP_MZN(rcpsp_model, cp_solver_name=CPSolverName.CHUFFED)
    parameters_cp = ParametersCP.default()
    parameters_cp.TimeLimit = 200
    result_storage = solver.solve(parameters_cp=parameters_cp, verbose=True)
    best = result_storage.get_best_solution_fit()
    print(best[1])
    pareto_store = result_storage_to_pareto_front(
        result_storage=result_storage, problem=rcpsp_model
    )
    # filename = 'j301_1.sm_postpro_dummy_cp.pkl'
    # import os, pickle
    # if os.path.exists(filename):
    #     os.remove(filename)
    # thefile = open(filename, 'ab')
    # pickle.dump(pareto_store, thefile)
    # thefile.close()
    # print(len(result_storage.list_solution_fits))
    # print(pareto_store.len_pareto_front())
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1)
    plot_storage_2d(
        result_storage=pareto_store,
        name_axis=["makespan", "mean_resource_reserve"],
        ax=ax,
    )
    plot_pareto_2d(
        pareto_front=pareto_store,
        name_axis=["makespan", "mean_resource_reserve"],
        ax=ax,
    )
    plt.show()


def rcpsp_cp_partial_solution():
    files_available = get_data_available()
    file = [f for f in files_available if "j601_2.sm" in f][0]
    rcpsp_problem = parse_file(file)
    solver = CP_RCPSP_MZN(rcpsp_problem)
    dummy_solution = rcpsp_problem.get_dummy_solution()
    some_constraints = {
        task: dummy_solution.rcpsp_schedule[task]["start_time"] + 5
        for task in [1, 2, 3, 4]
    }
    partial_solution = PartialSolution(
        task_mode=None, start_times=some_constraints
    )  # 4: 10})
    solver.init_model(partial_solution=partial_solution)
    parameters_cp = ParametersCP.default()
    result_storage = solver.solve(parameters_cp=parameters_cp, verbose=True)
    solution, fit = result_storage.get_best_solution_fit()
    print(solution)
    print("Constraint given as partial solution : ", partial_solution.start_times)
    print(
        "Found solution : ",
        {j: solution.rcpsp_schedule[j]["start_time"] for j in some_constraints},
    )
    print("Satisfy : ", rcpsp_problem.satisfy(solution))
    rcpsp_problem.plot_ressource_view(solution)
    import matplotlib.pyplot as plt

    plt.show()


if __name__ == "__main__":
    rcpsp_cp_partial_solution()
    # single_mode_rcpsp_cp()
    # single_mode_rcpsp_cp_intermediate_solution()
