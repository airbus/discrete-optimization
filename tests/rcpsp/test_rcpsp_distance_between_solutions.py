import numpy as np
from discrete_optimization.generic_tools.do_problem import ObjectiveHandling
from discrete_optimization.generic_tools.ea.ga import (
    DeapCrossover,
    DeapMutation,
    DeapSelection,
)
from discrete_optimization.generic_tools.ea.nsga import Nsga
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ParetoFront,
    plot_pareto_2d,
    plot_storage_2d,
)
from discrete_optimization.rcpsp.rcpsp_model import (
    MultiModeRCPSPModel,
    RCPSPModel,
    RCPSPSolution,
    SingleModeRCPSPModel,
    plt,
)
from discrete_optimization.rcpsp.rcpsp_parser import (
    files_available,
    get_data_available,
    parse_file,
)
from discrete_optimization.rcpsp.rcpsp_utils import (
    all_diff_start_time,
    kendall_tau_similarity,
)
from scipy.spatial import distance


def test_rank_solutions_by_permutation_distance():
    files = get_data_available()
    files = [f for f in files if "j301_1.sm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_model = parse_file(file_path)

    # Run NSGA to get ResultStorage

    mutation = DeapMutation.MUT_SHUFFLE_INDEXES
    objectives = ["makespan", "mean_resource_reserve"]
    objective_weights = [-1, +1]
    ga_solver = Nsga(
        rcpsp_model,
        encoding="rcpsp_permutation",
        objectives=objectives,
        objective_weights=objective_weights,
        mutation=mutation,
    )
    ga_solver._max_evals = 2000
    result_storage = ga_solver.solve()

    # Remove similar solutions from result storage
    result_storage.remove_duplicate_solutions(var_name="standardised_permutation")
    plot_storage_2d(result_storage=result_storage, name_axis=objectives)
    plt.show()

    # Pick one solution from the ResultStorage
    selected_index = 20
    selected_sol = result_storage.list_solution_fits[selected_index][0]

    # Calculate distance between selected solution and all solutions in resultStorage
    pop_ktds = {}
    for i in range(len(result_storage.list_solution_fits)):
        tmp_sol = result_storage.list_solution_fits[i][0]
        pop_ktds[i] = kendall_tau_similarity((selected_sol, tmp_sol))
    sorted_pop_ktds = {
        k: v
        for k, v in sorted(pop_ktds.items(), key=lambda item: item[1], reverse=True)
    }

    y_mean = []
    y_l1 = []
    y_l2 = []

    for i in range(len(result_storage.list_solution_fits)):
        key = list(sorted_pop_ktds.keys())[i]
        tmp_sol = result_storage.list_solution_fits[key][0]

        diffs = all_diff_start_time((selected_sol, tmp_sol))
        mean_diffs = np.mean([abs(diffs[key2]) for key2 in diffs.keys()])
        y_mean.append(mean_diffs)

        l1_dist = sum([abs(diffs[key2]) for key2 in diffs.keys()])
        y_l1.append(l1_dist)

        start_times_val_1 = [
            result_storage.list_solution_fits[selected_index][0].rcpsp_schedule[key2][
                "start_time"
            ]
            for key2 in result_storage.list_solution_fits[selected_index][
                0
            ].rcpsp_schedule.keys()
        ]
        start_times_val_2 = [
            result_storage.list_solution_fits[key][0].rcpsp_schedule[key2]["start_time"]
            for key2 in result_storage.list_solution_fits[key][0].rcpsp_schedule.keys()
        ]

        l2_dist = distance.euclidean(start_times_val_1, start_times_val_2)
        y_l2.append(l2_dist)

    # plot ktd vs. diffs in start time
    x = [sorted_pop_ktds[key] for key in sorted_pop_ktds.keys()]
    y_label = "Mean start time difference"
    y = y_mean
    plt.subplot(221)
    plt.scatter(x, y)
    # plt.title('Permutation vs. Schedule distances')
    plt.xlabel("KT permutation similarity")
    plt.ylabel(y_label)
    # plt.show()

    y_label = "L1 start time difference"
    y = y_l1
    plt.subplot(222)
    plt.scatter(x, y)
    # plt.title('Permutation vs. Schedule distances')
    plt.xlabel("KT permutation similarity")
    plt.ylabel(y_label)
    # plt.show()

    y_label = "L2 start time difference"
    y = y_l2
    plt.subplot(223)
    plt.scatter(x, y)
    # plt.title('Permutation vs. Schedule distances')
    plt.xlabel("KT permutation similarity")
    plt.ylabel(y_label)
    plt.show()


if __name__ == "__main__":
    test_rank_solutions_by_permutation_distance()