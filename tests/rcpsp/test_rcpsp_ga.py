import random

import pytest
from discrete_optimization.generic_tools.do_problem import ObjectiveHandling
from discrete_optimization.generic_tools.ea.alternating_ga import AlternatingGa
from discrete_optimization.generic_tools.ea.ga import (
    DeapCrossover,
    DeapMutation,
    DeapSelection,
    Ga,
)
from discrete_optimization.rcpsp.rcpsp_model import (
    MultiModeRCPSPModel,
    RCPSPModel,
    RCPSPSolution,
    SingleModeRCPSPModel,
    plt,
)
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file


@pytest.fixture
def random_seed():
    random.seed(0)


def test_single_mode_ga(random_seed):
    files = get_data_available()
    files = [f for f in files if "j1201_1.sm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_model = parse_file(file_path)

    mutation = DeapMutation.MUT_SHUFFLE_INDEXES
    ga_solver = Ga(
        rcpsp_model,
        encoding="rcpsp_permutation",
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=["makespan"],
        objective_weights=[-1],
        mutation=mutation,
    )
    ga_solver._max_evals = 10000
    sol = ga_solver.solve().get_best_solution()
    assert rcpsp_model.satisfy(sol)
    rcpsp_model.plot_ressource_view(sol)
    fitnesses = rcpsp_model.evaluate(sol)
    assert fitnesses == {"makespan": 122, "mean_resource_reserve": 0}


def test_multi_mode_alternating_ga(random_seed):
    files = get_data_available()
    files = [f for f in files if "j1010_5.mm" in f]  # Multi-mode RCPSP
    file_path = files[0]
    rcpsp_model = parse_file(file_path)

    total_evals = 10000
    number_of_meta_iterations = 5
    evals_per_ga_runs_perm = 0.5 * (total_evals / number_of_meta_iterations)
    evals_per_ga_runs_modes = 0.5 * (
        total_evals / number_of_meta_iterations
    )  # total_evals/(2*number_of_meta_iterations)

    mode_mutation = DeapMutation.MUT_UNIFORM_INT
    permutation_mutation = DeapMutation.MUT_SHUFFLE_INDEXES

    # Initialise the permutation that will be used to first search through the modes
    initial_permutation = [i for i in range(rcpsp_model.n_jobs)]
    rcpsp_model.set_fixed_permutation(initial_permutation)

    for it in range(number_of_meta_iterations):
        # Run a GA for evals_per_ga_runs evals on modes
        ga_solver = Ga(
            rcpsp_model,
            encoding="rcpsp_modes",
            objective_handling=ObjectiveHandling.AGGREGATE,
            objectives=["makespan"],
            objective_weights=[-1],
            mutation=mode_mutation,
            max_evals=evals_per_ga_runs_modes,
        )
        tmp_sol = ga_solver.solve().get_best_solution()
        print("best after modes search iteration: ", rcpsp_model.evaluate(tmp_sol))
        # Fix the resulting modes
        rcpsp_model.set_fixed_modes(tmp_sol.rcpsp_modes)

        # Run a GA for evals_per_ga_runs evals on permutation
        ga_solver = Ga(
            rcpsp_model,
            encoding="rcpsp_permutation",
            objective_handling=ObjectiveHandling.AGGREGATE,
            objectives=["makespan"],
            objective_weights=[-1],
            mutation=permutation_mutation,
            max_evals=evals_per_ga_runs_perm,
        )
        tmp_sol = ga_solver.solve().get_best_solution()
        print(
            "best after permutation search iteration: ", rcpsp_model.evaluate(tmp_sol)
        )

        # Fix the resulting permutation
        rcpsp_model.set_fixed_permutation(tmp_sol.rcpsp_permutation)

    sol = tmp_sol
    print(sol)
    fitnesses = rcpsp_model.evaluate(sol)
    print("fitnesses: ", fitnesses)


def test_multi_mode_alternating_ga_specific_mode_arrity(random_seed):

    files = get_data_available()
    files = [f for f in files if "j1010_10.mm" in f]  # Multi-mode RCPSP
    file_path = files[0]
    rcpsp_model: MultiModeRCPSPModel = parse_file(file_path)

    total_evals = 10000
    number_of_meta_iterations = 5
    evals_per_ga_runs_perm = 0.5 * (total_evals / number_of_meta_iterations)
    evals_per_ga_runs_modes = 0.5 * (
        total_evals / number_of_meta_iterations
    )  # total_evals/(2*number_of_meta_iterations)

    mode_mutation = DeapMutation.MUT_UNIFORM_INT
    permutation_mutation = DeapMutation.MUT_SHUFFLE_INDEXES

    # Initialise the permutation that will be used to first search through the modes
    initial_permutation = [i for i in range(rcpsp_model.n_jobs)]
    rcpsp_model.set_fixed_permutation(initial_permutation)

    for it in range(number_of_meta_iterations):
        # Run a GA for evals_per_ga_runs evals on modes
        ga_solver = Ga(
            rcpsp_model,
            encoding="rcpsp_modes_arrity_fix",
            objective_handling=ObjectiveHandling.AGGREGATE,
            objectives=["makespan"],
            objective_weights=[-1],
            mutation=mode_mutation,
            max_evals=evals_per_ga_runs_modes,
        )
        tmp_sol = ga_solver.solve().get_best_solution()
        print("best after modes search iteration: ", rcpsp_model.evaluate(tmp_sol))
        # Fix the resulting modes
        rcpsp_model.set_fixed_modes(tmp_sol.rcpsp_modes)

        # Run a GA for evals_per_ga_runs evals on permutation
        ga_solver = Ga(
            rcpsp_model,
            encoding="rcpsp_permutation",
            objective_handling=ObjectiveHandling.AGGREGATE,
            objectives=["makespan"],
            objective_weights=[-1],
            mutation=permutation_mutation,
            max_evals=evals_per_ga_runs_perm,
        )
        tmp_sol = ga_solver.solve().get_best_solution()
        print(
            "best after permutation search iteration: ", rcpsp_model.evaluate(tmp_sol)
        )

        # Fix the resulting permutation
        rcpsp_model.set_fixed_permutation(tmp_sol.rcpsp_permutation)

    sol = tmp_sol
    print(sol)
    fitnesses = rcpsp_model.evaluate(sol)
    print("fitnesses: ", fitnesses)


def test_alternating_ga_specific_mode_arrity_single_solver(random_seed):
    files = get_data_available()
    files = [f for f in files if "j1010_10.mm" in f]  # Multi-mode RCPSP
    file_path = files[0]
    rcpsp_model = parse_file(file_path)

    total_evals = 1000

    sub_evals = [50, 50]

    ga_solver = AlternatingGa(
        rcpsp_model,
        encodings=["rcpsp_modes_arrity_fix", "rcpsp_permutation"],
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=["makespan"],
        objective_weights=[-1],
        mutations=[DeapMutation.MUT_UNIFORM_INT, DeapMutation.MUT_SHUFFLE_INDEXES],
        crossovers=[DeapCrossover.CX_ONE_POINT, DeapCrossover.CX_PARTIALY_MATCHED],
        max_evals=total_evals,
        sub_evals=sub_evals,
    )

    tmp_sol = ga_solver.solve().get_best_solution()
    assert rcpsp_model.satisfy(tmp_sol)
    # assert rcpsp_model.evaluate(tmp_sol) == {'makespan': 36, 'mean_resource_reserve': 0}


def test_single_mode_moga_aggregated(random_seed):
    files = get_data_available()
    files = [f for f in files if "j301_1.sm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_model = parse_file(file_path)

    mutation = DeapMutation.MUT_SHUFFLE_INDEXES
    objectives = ["makespan", "mean_resource_reserve"]
    objective_weights = [-1, -100]
    ga_solver = Ga(
        rcpsp_model,
        encoding="rcpsp_permutation",
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=objectives,
        objective_weights=objective_weights,
        mutation=mutation,
    )
    ga_solver._max_evals = 2000
    sol = ga_solver.solve().get_best_solution()
    assert rcpsp_model.satisfy(sol)

    rcpsp_model.plot_ressource_view(sol)

    fitnesses = rcpsp_model.evaluate(sol)
    assert fitnesses == {"makespan": 43, "mean_resource_reserve": 0}

    mutation = DeapMutation.MUT_SHUFFLE_INDEXES
    objectives = ["makespan", "mean_resource_reserve"]
    objective_weights = [-1, +200]
    ga_solver = Ga(
        rcpsp_model,
        encoding="rcpsp_permutation",
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=objectives,
        objective_weights=objective_weights,
        mutation=mutation,
    )
    ga_solver._max_evals = 2000
    sol = ga_solver.solve().get_best_solution()
    assert rcpsp_model.satisfy(sol)

    rcpsp_model.plot_ressource_view(sol)
    fitnesses = rcpsp_model.evaluate(sol)
    assert fitnesses == {"makespan": 43, "mean_resource_reserve": 0}


def test_own_pop_single_mode_ga(random_seed):
    files = get_data_available()
    files = [f for f in files if "j301_1.sm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_model = parse_file(file_path)

    mutation = DeapMutation.MUT_SHUFFLE_INDEXES

    initial_population = []
    for i in range(5):
        ind = [x for x in range(rcpsp_model.n_jobs_non_dummy)]
        initial_population.append(ind)

    ga_solver = Ga(
        rcpsp_model,
        encoding="rcpsp_permutation",
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=["makespan"],
        objective_weights=[-1],
        mutation=mutation,
        initial_population=initial_population,
    )
    ga_solver._max_evals = 100
    sol = ga_solver.solve().get_best_solution()
    assert rcpsp_model.satisfy(sol)

    rcpsp_model.plot_ressource_view(sol)

    fitnesses = rcpsp_model.evaluate(sol)
    assert fitnesses == {"makespan": 49, "mean_resource_reserve": 0}


if __name__ == "__main__":
    test_own_pop_single_mode_ga()
