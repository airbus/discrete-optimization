#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random

import numpy as np
import pytest

from discrete_optimization.generic_tools.do_problem import ObjectiveHandling
from discrete_optimization.generic_tools.ea.alternating_ga import AlternatingGa
from discrete_optimization.generic_tools.ea.ga import DeapCrossover, DeapMutation, Ga
from discrete_optimization.generic_tools.mutations.mixed_mutation import (
    BasicPortfolioMutation,
)
from discrete_optimization.generic_tools.mutations.mutation_catalog import (
    get_available_mutations,
)
from discrete_optimization.rcpsp.mutation import PermutationMutationRcpsp
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solution import RcpspSolution
from discrete_optimization.rcpsp.solvers.cpm import CpmRcpspSolver


@pytest.fixture
def random_seed():
    random.seed(0)
    np.random.seed(0)


def test_single_mode_ga(random_seed):
    files = get_data_available()
    files = [f for f in files if "j1201_1.sm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_problem = parse_file(file_path)

    mutation = DeapMutation.MUT_SHUFFLE_INDEXES
    ga_solver = Ga(
        rcpsp_problem,
        encoding="rcpsp_permutation",
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=["makespan"],
        objective_weights=[-1],
        mutation=mutation,
    )
    ga_solver._max_evals = 10000
    sol = ga_solver.solve().get_best_solution()
    assert rcpsp_problem.satisfy(sol)
    rcpsp_problem.plot_ressource_view(sol)
    fitnesses = rcpsp_problem.evaluate(sol)


def test_multi_mode_alternating_ga(random_seed):
    files = get_data_available()
    files = [f for f in files if "j1010_5.mm" in f]  # Multi-mode RCPSP
    file_path = files[0]
    rcpsp_problem = parse_file(file_path)

    total_evals = 10000
    number_of_meta_iterations = 5
    evals_per_ga_runs_perm = int(0.5 * (total_evals / number_of_meta_iterations))
    evals_per_ga_runs_modes = int(0.5 * (total_evals / number_of_meta_iterations))

    mode_mutation = DeapMutation.MUT_UNIFORM_INT
    permutation_mutation = DeapMutation.MUT_SHUFFLE_INDEXES

    # Initialise the permutation that will be used to first search through the modes
    initial_permutation = [i for i in range(rcpsp_problem.n_jobs_non_dummy)]
    rcpsp_problem.set_fixed_permutation(initial_permutation)
    tmp_sol = None
    for it in range(number_of_meta_iterations):
        # Run a GA for evals_per_ga_runs evals on modes
        ga_solver = Ga(
            rcpsp_problem,
            encoding="rcpsp_modes",
            objective_handling=ObjectiveHandling.AGGREGATE,
            objectives=["makespan"],
            objective_weights=[-1],
            mutation=mode_mutation,
            max_evals=evals_per_ga_runs_modes,
        )
        result = ga_solver.solve()
        tmp_sol, fit = result.get_best_solution_fit()
        # Fix the resulting modes
        rcpsp_problem.set_fixed_modes(tmp_sol.rcpsp_modes)

        # Run a GA for evals_per_ga_runs evals on permutation
        ga_solver = Ga(
            rcpsp_problem,
            encoding="rcpsp_permutation",
            objective_handling=ObjectiveHandling.AGGREGATE,
            objectives=["makespan"],
            objective_weights=[-1],
            mutation=permutation_mutation,
            max_evals=evals_per_ga_runs_perm,
        )
        tmp_sol = ga_solver.solve().get_best_solution()

        # Fix the resulting permutation
        rcpsp_problem.set_fixed_permutation(tmp_sol.rcpsp_permutation)
    sol = tmp_sol
    fitnesses = rcpsp_problem.evaluate(sol)


def test_multi_mode_alternating_ga_specific_mode_arity(random_seed):

    files = get_data_available()
    files = [f for f in files if "j1010_10.mm" in f]  # Multi-mode RCPSP
    file_path = files[0]
    rcpsp_problem = parse_file(file_path)

    total_evals = 10000
    number_of_meta_iterations = 5
    evals_per_ga_runs_perm = int(0.5 * (total_evals / number_of_meta_iterations))
    evals_per_ga_runs_modes = int(0.5 * (total_evals / number_of_meta_iterations))

    mode_mutation = DeapMutation.MUT_UNIFORM_INT
    permutation_mutation = DeapMutation.MUT_SHUFFLE_INDEXES

    # Initialise the permutation that will be used to first search through the modes
    initial_permutation = [i for i in range(rcpsp_problem.n_jobs_non_dummy)]
    rcpsp_problem.set_fixed_permutation(initial_permutation)

    for it in range(number_of_meta_iterations):
        # Run a GA for evals_per_ga_runs evals on modes
        ga_solver = Ga(
            rcpsp_problem,
            encoding="rcpsp_modes_arity_fix",
            objective_handling=ObjectiveHandling.AGGREGATE,
            objectives=["makespan"],
            objective_weights=[-1],
            mutation=mode_mutation,
            max_evals=evals_per_ga_runs_modes,
        )
        tmp_sol = ga_solver.solve().get_best_solution()
        # Fix the resulting modes
        rcpsp_problem.set_fixed_modes(tmp_sol.rcpsp_modes)

        # Run a GA for evals_per_ga_runs evals on permutation
        ga_solver = Ga(
            rcpsp_problem,
            encoding="rcpsp_permutation",
            objective_handling=ObjectiveHandling.AGGREGATE,
            objectives=["makespan"],
            objective_weights=[-1],
            mutation=permutation_mutation,
            max_evals=evals_per_ga_runs_perm,
        )
        tmp_sol = ga_solver.solve().get_best_solution()

        # Fix the resulting permutation
        rcpsp_problem.set_fixed_permutation(tmp_sol.rcpsp_permutation)

    sol = tmp_sol
    fitnesses = rcpsp_problem.evaluate(sol)


def test_alternating_ga_specific_mode_arity_single_solver(random_seed):
    files = get_data_available()
    files = [f for f in files if "j1010_10.mm" in f]  # Multi-mode RCPSP
    file_path = files[0]
    rcpsp_problem = parse_file(file_path)

    total_evals = 1000

    sub_evals = [50, 50]

    ga_solver = AlternatingGa(
        rcpsp_problem,
        encodings=["rcpsp_modes_arity_fix", "rcpsp_permutation"],
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=["makespan"],
        objective_weights=[-1],
        mutations=[DeapMutation.MUT_UNIFORM_INT, DeapMutation.MUT_SHUFFLE_INDEXES],
        crossovers=[DeapCrossover.CX_ONE_POINT, DeapCrossover.CX_PARTIALY_MATCHED],
        max_evals=total_evals,
        sub_evals=sub_evals,
    )

    tmp_sol, fit = ga_solver.solve().get_best_solution_fit()
    assert rcpsp_problem.satisfy(tmp_sol)
    print(fit)


def test_alternating_ga_specific_mode_arity_single_solver_warm_started_with_cpm(
    random_seed,
):
    files = get_data_available()
    files = [f for f in files if "j1010_10.mm" in f]  # Multi-mode RCPSP
    file_path = files[0]
    rcpsp_problem = parse_file(file_path)

    cpm = CpmRcpspSolver(problem=rcpsp_problem)
    cpath = cpm.run_classic_cpm()
    order = cpm.return_order_cpm()
    permutation_sgs = [o - 2 for o in order]
    permutation_sgs.remove(min(permutation_sgs))
    permutation_sgs.remove(max(permutation_sgs))
    solution_sgs = RcpspSolution(
        problem=rcpsp_problem,
        rcpsp_permutation=permutation_sgs,
        rcpsp_modes=[1 for i in range(rcpsp_problem.n_jobs_non_dummy)],
    )
    fit_sgs = cpm.aggreg_from_sol(solution_sgs)

    print(f"SGS: {fit_sgs}")

    total_evals = 1000

    sub_evals = [50, 50]

    ga_solver = AlternatingGa(
        rcpsp_problem,
        encodings=["rcpsp_modes_arity_fix", "rcpsp_permutation"],
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=["makespan"],
        objective_weights=[-1],
        mutations=[DeapMutation.MUT_UNIFORM_INT, DeapMutation.MUT_SHUFFLE_INDEXES],
        crossovers=[DeapCrossover.CX_ONE_POINT, DeapCrossover.CX_PARTIALY_MATCHED],
        max_evals=total_evals,
        sub_evals=sub_evals,
    )
    ga_solver.set_warm_start(solution_sgs)
    tmp_sol, fit_ga = ga_solver.solve().get_best_solution_fit()
    assert rcpsp_problem.satisfy(tmp_sol)

    print(f"GA: {fit_ga}")

    assert fit_ga > fit_sgs


def test_single_mode_moga_aggregated(random_seed):
    files = get_data_available()
    files = [f for f in files if "j301_1.sm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_problem = parse_file(file_path)

    mutation = DeapMutation.MUT_SHUFFLE_INDEXES
    objectives = ["makespan", "mean_resource_reserve"]
    objective_weights = [-1, -100]
    ga_solver = Ga(
        rcpsp_problem,
        encoding="rcpsp_permutation",
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=objectives,
        objective_weights=objective_weights,
        mutation=mutation,
    )
    ga_solver._max_evals = 2000
    sol = ga_solver.solve().get_best_solution()
    assert rcpsp_problem.satisfy(sol)

    rcpsp_problem.plot_ressource_view(sol)

    fitnesses = rcpsp_problem.evaluate(sol)
    assert fitnesses == {
        "makespan": 43,
        "mean_resource_reserve": 0,
        "constraint_penalty": 0.0,
    }

    mutation = DeapMutation.MUT_SHUFFLE_INDEXES
    objectives = ["makespan", "mean_resource_reserve"]
    objective_weights = [-1, +200]
    ga_solver = Ga(
        rcpsp_problem,
        encoding="rcpsp_permutation",
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=objectives,
        objective_weights=objective_weights,
        mutation=mutation,
    )
    ga_solver._max_evals = 2000
    sol = ga_solver.solve().get_best_solution()
    assert rcpsp_problem.satisfy(sol)

    rcpsp_problem.plot_ressource_view(sol)
    fitnesses = rcpsp_problem.evaluate(sol)
    assert fitnesses == {
        "makespan": 43,
        "mean_resource_reserve": 0,
        "constraint_penalty": 0.0,
    }


def test_own_pop_single_mode_ga(random_seed):
    files = get_data_available()
    files = [f for f in files if "j301_1.sm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_problem = parse_file(file_path)

    mutation = DeapMutation.MUT_SHUFFLE_INDEXES

    initial_population = []
    for i in range(5):
        ind = [x for x in range(rcpsp_problem.n_jobs_non_dummy)]
        initial_population.append(ind)

    ga_solver = Ga(
        rcpsp_problem,
        encoding="rcpsp_permutation",
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=["makespan"],
        objective_weights=[-1],
        mutation=mutation,
        initial_population=initial_population,
    )
    ga_solver._max_evals = 100
    sol = ga_solver.solve().get_best_solution()
    assert rcpsp_problem.satisfy(sol)

    rcpsp_problem.plot_ressource_view(sol)

    fitnesses = rcpsp_problem.evaluate(sol)
    assert fitnesses == {
        "makespan": 49,
        "mean_resource_reserve": 0,
        "constraint_penalty": 0.0,
    }


def test_ga_warm_started_with_cpm(random_seed):
    files_available = get_data_available()
    file = [f for f in files_available if "j1201_1.sm" in f][0]
    rcpsp_problem = parse_file(file)
    dummy = rcpsp_problem.get_dummy_solution()

    cpm = CpmRcpspSolver(problem=rcpsp_problem)
    cpath = cpm.run_classic_cpm()
    order = cpm.return_order_cpm()
    permutation_sgs = [o - 2 for o in order]
    permutation_sgs.remove(min(permutation_sgs))
    permutation_sgs.remove(max(permutation_sgs))
    solution_sgs = RcpspSolution(
        problem=rcpsp_problem,
        rcpsp_permutation=permutation_sgs,
        rcpsp_modes=[1 for i in range(rcpsp_problem.n_jobs_non_dummy)],
    )
    fit_sgs = cpm.aggreg_from_sol(solution_sgs)

    print(f"SGS: {fit_sgs}")

    _, mutations = get_available_mutations(rcpsp_problem, dummy)
    list_mutation = [
        mutate[0].build(rcpsp_problem, dummy, **mutate[1])
        for mutate in mutations
        if mutate[0] == PermutationMutationRcpsp
    ]
    mixed_mutation = BasicPortfolioMutation(
        list_mutation, np.ones((len(list_mutation)))
    )
    kwargs = dict(
        problem=rcpsp_problem,
        objectives=["makespan"],
        objective_weights=[-1.0],
        objective_handling=ObjectiveHandling.AGGREGATE,
        mutation=mixed_mutation,
        max_evals=5000,
    )
    ga_solver = Ga(**kwargs)
    ga_solver.set_warm_start(solution_sgs)
    result_store = ga_solver.solve()
    solution, fit_ga = result_store.get_best_solution_fit()
    assert rcpsp_problem.satisfy(solution)
    print(f"GA: {fit_ga}")

    assert fit_ga > fit_sgs


if __name__ == "__main__":
    test_own_pop_single_mode_ga()
