#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from pytest_cases import fixture

from discrete_optimization.facility.problem import FacilitySolution
from discrete_optimization.facility.solvers.cpsat import CpSatFacilitySolver
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import SignEnum


@fixture
def solver(problem):
    solver = CpSatFacilitySolver(problem=problem)
    solver.init_model()
    return solver


def test_solve(problem, solver):
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert problem.satisfy(sol)


def test_warm_start(problem, solver, start_solution):
    sol: FacilitySolution = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert problem.satisfy(sol)
    assert sol.facility_for_customers != start_solution.facility_for_customers
    print(len(set(sol.facility_for_customers)))

    solver.set_warm_start(start_solution)
    sol: FacilitySolution = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert problem.satisfy(sol)
    assert sol.facility_for_customers == start_solution.facility_for_customers

    print(len(set(sol.facility_for_customers)))
    print(sol.facility_for_customers)

    print(len(problem.unary_resources_list))


def test_add_constraint_on_nb_allocation_changes(solver, problem, start_solution):
    nb_changes = 2
    sol: FacilitySolution
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
