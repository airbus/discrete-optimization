from discrete_optimization.rcpsp.rcpsp_model import (
    MethodBaseRobustification,
    MethodRobustification,
    MultiModeRCPSPModel,
    RCPSPModel,
    RCPSPSolution,
    SingleModeRCPSPModel,
    UncertainRCPSPModel,
    create_poisson_laws_duration,
    plt,
)
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file
from discrete_optimization.rcpsp.rcpsp_utils import (
    plot_resource_individual_gantt,
    plot_ressource_view,
)
from discrete_optimization.rcpsp.solver.rcpsp_pile import GreedyChoice, PileSolverRCPSP


def run_pile():
    files = get_data_available()
    files = [f for f in files if "j1201_1.sm" in f]  #
    file_path = files[0]
    rcpsp_model = parse_file(file_path)
    solver = PileSolverRCPSP(rcpsp_model=rcpsp_model)
    fits = []
    for k in range(10):
        result_storage = solver.solve(greedy_choice=GreedyChoice.SAMPLE_MOST_SUCCESSORS)
        sol, fit = result_storage.get_best_solution_fit()
        print(sol, fit)
        print(rcpsp_model.satisfy(sol))
        fits += [fit]
    sol_2 = RCPSPSolution(problem=rcpsp_model, rcpsp_permutation=sol.rcpsp_permutation)
    print("Permutation : ", sol.rcpsp_permutation)
    fit = rcpsp_model.evaluate(sol_2)
    print("Schedule from greedy ", sol.rcpsp_schedule)
    print("recomputed : ", fit)
    print("satisfy sol2", rcpsp_model.satisfy(sol_2))
    print("Recomputed schedule from permutation : ", sol_2.rcpsp_schedule)
    plot_ressource_view(rcpsp_model, sol)
    plot_ressource_view(rcpsp_model, sol_2)
    plot_resource_individual_gantt(rcpsp_model, sol)
    plt.show()
    print(rcpsp_model.satisfy(sol))


def run_pile_multimode():
    files = get_data_available()
    files = [f for f in files if "j1010_1.mm" in f]  #
    file_path = files[0]
    rcpsp_model = parse_file(file_path)
    solver = PileSolverRCPSP(rcpsp_model=rcpsp_model)
    fits = []
    for k in range(10):
        result_storage = solver.solve(greedy_choice=GreedyChoice.SAMPLE_MOST_SUCCESSORS)
        sol, fit = result_storage.get_best_solution_fit()
        print(sol, fit)
        print(rcpsp_model.satisfy(sol))
        fits += [fit]
    sol_2 = RCPSPSolution(
        problem=rcpsp_model,
        rcpsp_modes=sol.rcpsp_modes,
        rcpsp_permutation=sol.rcpsp_permutation,
    )
    print("Permutation : ", sol.rcpsp_permutation)
    fit = rcpsp_model.evaluate(sol_2)
    print("Schedule from greedy ", sol.rcpsp_schedule)
    print("recomputed : ", fit)
    print("satisfy sol2", rcpsp_model.satisfy(sol_2))
    print("Recomputed schedule from permutation : ", sol_2.rcpsp_schedule)
    plot_ressource_view(rcpsp_model, sol)
    plot_ressource_view(rcpsp_model, sol_2)
    plot_resource_individual_gantt(rcpsp_model, sol)
    plt.show()
    print(rcpsp_model.satisfy(sol))


def run_pile_robust():
    files = get_data_available()
    files = [f for f in files if "j1201_1.sm" in f]  #
    file_path = files[0]
    rcpsp_model: RCPSPModel = parse_file(file_path)
    poisson_laws = create_poisson_laws_duration(rcpsp_model)
    uncertain = UncertainRCPSPModel(rcpsp_model, poisson_laws=poisson_laws)
    worst = uncertain.create_rcpsp_model(
        MethodRobustification(MethodBaseRobustification.WORST_CASE, percentile=0)
    )
    solver = PileSolverRCPSP(rcpsp_model=worst)
    solver_original = PileSolverRCPSP(rcpsp_model=rcpsp_model)
    sol_origin, fit = solver_original.solve(greedy_choice=GreedyChoice.MOST_SUCCESSORS)
    print(fit, "fitness found on original case")
    sol, fit = solver.solve(greedy_choice=GreedyChoice.MOST_SUCCESSORS)
    print(fit, "fitness found on worst case")
    many_random_instance = [
        uncertain.create_rcpsp_model(
            method_robustification=MethodRobustification(
                MethodBaseRobustification.SAMPLE
            )
        )
        for i in range(1000)
    ]
    many_random_instance = []
    many_random_instance += [
        uncertain.create_rcpsp_model(
            method_robustification=MethodRobustification(
                MethodBaseRobustification.PERCENTILE, percentile=j
            )
        )
        for j in range(80, 100)
    ]
    permutation = sol.rcpsp_permutation
    permutation_original = sol_origin.rcpsp_permutation
    fits = []
    fits_original = []
    for instance in many_random_instance:
        sol_ = RCPSPSolution(problem=instance, rcpsp_permutation=permutation)
        fit = instance.evaluate(sol_)
        fits += [fit]
        sol_ = RCPSPSolution(problem=instance, rcpsp_permutation=permutation_original)
        fit = instance.evaluate(sol_)
        fits_original += [fit]

    sol_ = RCPSPSolution(problem=rcpsp_model, rcpsp_permutation=permutation)
    fit = rcpsp_model.evaluate(sol_)
    print("Fit on original problem worst :", fit)

    sol_ = RCPSPSolution(problem=rcpsp_model, rcpsp_permutation=permutation_original)
    fit = rcpsp_model.evaluate(sol_)
    print("Fit on original problem origin :", fit)
    import numpy as np
    import scipy.stats
    import seaborn as sns

    makespans = np.array([f["makespan"] for f in fits])
    makespans_origin = np.array([f["makespan"] for f in fits_original])

    print("Stats from robust version : ", scipy.stats.describe(makespans))
    print(min(makespans), max(makespans))
    print("Stats from original model", scipy.stats.describe(makespans_origin))
    print(min(makespans_origin), max(makespans_origin))
    import matplotlib.pyplot as plt

    sns.distplot(
        makespans, rug=True, bins=len(many_random_instance) // 10, label="Robust"
    )
    sns.distplot(
        makespans_origin,
        rug=True,
        bins=len(many_random_instance) // 10,
        label="Original",
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_pile_multimode()
