#  Copyright (c) 2024-2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import pytest
from pytest_cases import fixture

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import SignEnum
from discrete_optimization.knapsack.problem import KnapsackSolution
from discrete_optimization.knapsack.solvers.cpsat import CpSatKnapsackSolver


@fixture
def solver(problem):
    solver = CpSatKnapsackSolver(problem)
    solver.init_model()
    return solver


def test_cp_knapsack(problem, solver, start_solution):
    result_storage = solver.solve(time_limit=10)
    sol: KnapsackSolution
    sol, fit = result_storage.get_best_solution_fit()
    assert problem.satisfy(sol)

    # check is_allocated
    for task in problem.tasks_list:
        for unary_resource in problem.unary_resources_list:
            sol.is_allocated(task, unary_resource)
            solver.get_task_unary_resource_is_present_variable(task, unary_resource)
        assert not sol.is_allocated(task, False)
        with pytest.raises(ValueError):
            solver.get_task_unary_resource_is_present_variable(task, False)

    # first solution is not start_solution
    assert result_storage[0][0].list_taken != start_solution.list_taken

    # warm start at first solution
    solver.set_warm_start(start_solution)
    # force first solution to be the hinted one
    result_storage = solver.solve(
        time_limit=10,
        ortools_cpsat_solver_kwargs=dict(fix_variables_to_their_hinted_value=True),
    )
    assert result_storage[0][0].list_taken == start_solution.list_taken


def test_constraint_nb_allocation_changes(problem, solver, start_solution):
    nb_changes = 3
    sol: KnapsackSolution
    ref = start_solution

    # force to be away from start_solution
    constraints = solver.add_constraint_on_nb_allocation_changes(
        ref=ref, nb_changes=nb_changes, sign=SignEnum.UP
    )
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert (
        sol.compute_nb_allocation_changes(
            ref,
            tasks=solver.subset_tasks_of_interest,
            unary_resources=solver.subset_unaryresources_allowed,
        )
        > nb_changes
    )

    solver.remove_constraints(constraints)

    # force to be close to start_solution
    solver.add_constraint_on_nb_allocation_changes(ref=ref, nb_changes=nb_changes)
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert (
        sol.compute_nb_allocation_changes(
            ref,
            tasks=solver.subset_tasks_of_interest,
            unary_resources=solver.subset_unaryresources_allowed,
        )
        <= nb_changes
    )


def test_nb_tasks_done(problem, solver):
    sol: KnapsackSolution
    var = solver.get_nb_tasks_done_variable()
    nb_tasks = 2

    constraints = solver.add_bound_constraint(var, SignEnum.LEQ, nb_tasks)
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert sum(sol.list_taken) <= nb_tasks

    solver.remove_constraints(constraints)
    constraints = solver.add_bound_constraint(var, SignEnum.UP, nb_tasks)
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert sum(sol.list_taken) > nb_tasks
