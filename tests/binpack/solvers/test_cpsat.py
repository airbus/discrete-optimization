#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import pytest
from pytest_cases import fixture, param_fixture

from discrete_optimization.binpack.problem import BinPackSolution
from discrete_optimization.binpack.solvers.cpsat import (
    CpSatBinPackSolver,
    ModelingBinPack,
    ModelingError,
)
from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import SignEnum

modeling = param_fixture(
    "modeling",
    [ModelingBinPack.BINARY, ModelingBinPack.SCHEDULING],
)


@fixture
def solver(problem, modeling):
    solver = CpSatBinPackSolver(problem=problem)
    solver.init_model(upper_bound=20, modeling=modeling)
    return solver


@fixture
def solver_kwargs():
    return dict(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    )


@fixture
def solver_binary(problem):
    solver = CpSatBinPackSolver(problem=problem)
    solver.init_model(upper_bound=20, modeling=ModelingBinPack.BINARY)
    return solver


@fixture
def solver_scheduling(problem):
    solver = CpSatBinPackSolver(problem=problem)
    solver.init_model(upper_bound=20, modeling=ModelingBinPack.SCHEDULING)
    return solver


def test_cpsat(problem, solver, manual_sol, manual_sol2):
    solve_kwargs = dict(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    )
    res = solver.solve(**solve_kwargs)
    sol = res[-1][0]
    assert problem.satisfy(sol)

    # check warm start
    if manual_sol.allocation == sol.allocation:
        # ensure using different sol as warm start
        manual_sol = manual_sol2
    assert manual_sol.allocation != sol.allocation
    solver.set_warm_start(manual_sol)
    res = solver.solve(**solve_kwargs)
    sol = res[-1][0]
    assert problem.satisfy(sol)
    assert manual_sol.allocation == sol.allocation


def test_binary_var_with_scheduling_modeling_nok(solver_scheduling):
    with pytest.raises(ModelingError):
        solver_scheduling.get_task_unary_resource_is_present_variable(0, 0)


def test_constraint_nb_allocation_changes(problem, solver_binary, manual_sol):
    nb_changes = 3
    sol: BinPackSolution
    ref = manual_sol
    solver = solver_binary

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


def test_scheduling_var_with_binary_modeling_nok(solver_binary):
    with pytest.raises(ModelingError):
        solver_binary.get_task_start_or_end_variable(0, StartOrEnd.START)


def test_task_constraint(problem, solver_scheduling):
    task = 2
    start_or_end = StartOrEnd.END
    sign = SignEnum.LEQ
    antisign = SignEnum.UP
    time = 5
    solver = solver_scheduling

    # anti-constraint
    cstrs = solver.add_constraint_on_task(
        task=task, start_or_end=start_or_end, sign=antisign, time=time
    )
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    ).get_best_solution()
    assert sol.constraint_on_task_satisfied(
        task=task, start_or_end=start_or_end, sign=antisign, time=time
    )
    # constraint
    solver.remove_constraints(cstrs)
    cstrs = solver.add_constraint_on_task(
        task=task, start_or_end=start_or_end, sign=sign, time=time
    )
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    ).get_best_solution()
    assert sol.constraint_on_task_satisfied(
        task=task, start_or_end=start_or_end, sign=sign, time=time
    )


def test_chaining_constraint(solver_scheduling):
    task1 = 7
    task2 = 8
    solver = solver_scheduling

    # before adding the constraint, not already satisfied
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert not sol.constraint_chaining_tasks_satisfied(task1=task1, task2=task2)
    # add constraint
    cstrs = solver.add_constraint_chaining_tasks(task1=task1, task2=task2)
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert sol.constraint_chaining_tasks_satisfied(task1=task1, task2=task2)
    # remove constraint
    solver.remove_constraints(cstrs)
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert not sol.constraint_chaining_tasks_satisfied(task1=task1, task2=task2)
