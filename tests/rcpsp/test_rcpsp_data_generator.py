from discrete_optimization.generic_tools.do_problem import ObjectiveHandling
from discrete_optimization.generic_tools.ea.ga import (
    DeapCrossover,
    DeapMutation,
    DeapSelection,
    Ga,
)
from discrete_optimization.generic_tools.ea.nsga import Nsga
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ParetoFront,
    plot_pareto_2d,
    plot_storage_2d,
)
from discrete_optimization.rcpsp.rcpsp_data_generator import (
    generate_rcpsp_with_helper_tasks_data,
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


def test_generate_rcpsp_with_helper_tasks_data():
    # Load initial RCPSP model
    files = get_data_available()
    files = [f for f in files if "j301_1.sm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_model = parse_file(file_path)

    # Some settings
    original_duration_multiplier = 1
    n_assisted_activities = 30
    n_assistants = 1
    probability_of_cross_helper_precedences = 0.5
    fixed_helper_duration = 2
    random_seed = 42

    rcpsp_h = generate_rcpsp_with_helper_tasks_data(
        rcpsp_model,
        original_duration_multiplier,
        n_assisted_activities,
        n_assistants,
        probability_of_cross_helper_precedences,
        fixed_helper_duration=fixed_helper_duration,
        random_seed=random_seed,
    )


def test_generate_and_solve_rcpsp_with_helper_tasks_data_makespan():
    # Load initial RCPSP model
    files = get_data_available()
    files = [f for f in files if "j301_1.sm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_model = parse_file(file_path)

    # Some settings
    original_duration_multiplier = 1
    n_assisted_activities = 30
    n_assistants = 1
    probability_of_cross_helper_precedences = 0.5
    fixed_helper_duration = 2
    random_seed = 42

    rcpsp_h = generate_rcpsp_with_helper_tasks_data(
        rcpsp_model,
        original_duration_multiplier,
        n_assisted_activities,
        n_assistants,
        probability_of_cross_helper_precedences,
        fixed_helper_duration=fixed_helper_duration,
        random_seed=random_seed,
    )

    ga_solver = Ga(
        rcpsp_h,
        encoding="rcpsp_permutation",
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=["makespan"],
        objective_weights=[-1],
        max_evals=500,
    )

    sol = ga_solver.solve()
    print(type(sol))
    print(sol)
    print(rcpsp_h.satisfy(sol))

    fitnesses = rcpsp_h.evaluate(sol)
    print("fitnesses: ", fitnesses)

    rcpsp_h.plot_ressource_view(sol)
    plt.show()


def test_generate_and_solve_rcpsp_with_helper_tasks_data_cumulated_helper_gap():
    # Load initial RCPSP model
    files = get_data_available()
    files = [f for f in files if "j301_1.sm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_model = parse_file(file_path)

    # Some settings
    original_duration_multiplier = 1
    n_assisted_activities = 30
    n_assistants = 1
    probability_of_cross_helper_precedences = 0.5
    fixed_helper_duration = 2
    random_seed = 42

    rcpsp_h = generate_rcpsp_with_helper_tasks_data(
        rcpsp_model,
        original_duration_multiplier,
        n_assisted_activities,
        n_assistants,
        probability_of_cross_helper_precedences,
        fixed_helper_duration=fixed_helper_duration,
        random_seed=random_seed,
    )

    ga_solver = Ga(
        rcpsp_h,
        encoding="rcpsp_permutation",
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=["cumulated_helper_gap"],
        objective_weights=[-1],
        max_evals=500,
    )

    sol = ga_solver.solve()
    print(type(sol))
    print(sol)
    print(rcpsp_h.satisfy(sol))

    fitnesses = rcpsp_h.evaluate(sol)
    print("fitnesses: ", fitnesses)

    rcpsp_h.plot_ressource_view(sol)
    plt.show()


def test_generate_and_solve_rcpsp_with_helper_tasks_data_nsga_3d():
    files = get_data_available()
    files = [f for f in files if "j301_1.sm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_model = parse_file(file_path)

    # Some settings
    original_duration_multiplier = 1
    n_assisted_activities = 30
    n_assistants = 1
    probability_of_cross_helper_precedences = 0.5
    fixed_helper_duration = 2
    random_seed = 42

    rcpsp_h = generate_rcpsp_with_helper_tasks_data(
        rcpsp_model,
        original_duration_multiplier,
        n_assisted_activities,
        n_assistants,
        probability_of_cross_helper_precedences,
        fixed_helper_duration=fixed_helper_duration,
        random_seed=random_seed,
    )

    mutation = DeapMutation.MUT_SHUFFLE_INDEXES
    objectives = ["makespan", "mean_resource_reserve", "cumulated_helper_gap"]
    objective_weights = [-1, 1, -1]
    ga_solver = Nsga(
        rcpsp_h,
        encoding="rcpsp_permutation",
        objectives=objectives,
        objective_weights=objective_weights,
        mutation=mutation,
    )
    ga_solver._max_evals = 200
    result_storage = ga_solver.solve()
    print(result_storage)

    # pareto_front = ParetoFront(result_storage)
    # print('pareto_front: ', pareto_front)

    # plot_pareto_2d(result_storage, name_axis=objectives)
    plot_storage_2d(result_storage=result_storage, name_axis=objectives)
    plt.show()


def test_generate_and_solve_rcpsp_with_helper_tasks_data_nsga_2d():
    files = get_data_available()
    files = [f for f in files if "j301_1.sm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_model = parse_file(file_path)

    # Some settings
    original_duration_multiplier = 1
    n_assisted_activities = 30
    n_assistants = 1
    probability_of_cross_helper_precedences = 0.5
    fixed_helper_duration = 2
    random_seed = 4
    rcpsp_h = generate_rcpsp_with_helper_tasks_data(
        rcpsp_model,
        original_duration_multiplier,
        n_assisted_activities,
        n_assistants,
        probability_of_cross_helper_precedences,
        fixed_helper_duration=fixed_helper_duration,
        random_seed=random_seed,
    )

    mutation = DeapMutation.MUT_SHUFFLE_INDEXES
    objectives = ["makespan", "cumulated_helper_gap"]
    objective_weights = [-1, -1]
    ga_solver = Nsga(
        rcpsp_h,
        encoding="rcpsp_permutation",
        objectives=objectives,
        objective_weights=objective_weights,
        mutation=mutation,
    )
    ga_solver._max_evals = 500
    result_storage = ga_solver.solve()
    print(result_storage)

    # pareto_front = ParetoFront(result_storage)
    # print('pareto_front: ', pareto_front)

    # plot_pareto_2d(result_storage, name_axis=objectives)
    plot_storage_2d(result_storage=result_storage, name_axis=objectives)
    plt.show()


if __name__ == "__main__":
    # test_generate_rcpsp_with_helper_tasks_data()
    # test_generate_and_solve_rcpsp_with_helper_tasks_data_makespan()
    # test_generate_and_solve_rcpsp_with_helper_tasks_data_cumulated_helper_gap()
    # test_generate_and_solve_rcpsp_with_helper_tasks_data_nsga_3d()
    test_generate_and_solve_rcpsp_with_helper_tasks_data_nsga_2d()
