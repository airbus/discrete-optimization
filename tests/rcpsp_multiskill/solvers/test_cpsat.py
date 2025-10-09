#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  In this example script we're parsing dzn files that are retrieved from https://github.com/youngkd/MSPSP-InstLib
#  And run CP solver with different mzn models.

import random

import numpy as np
import pytest

from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp, SignEnum
from discrete_optimization.rcpsp_multiskill.parser_imopse import (
    get_data_available,
    parse_file,
)
from discrete_optimization.rcpsp_multiskill.problem import MultiskillRcpspSolution
from discrete_optimization.rcpsp_multiskill.solvers.cpsat import (
    CpSatMultiskillRcpspSolver,
)


def test_imopse_cpsat():
    file = [f for f in get_data_available() if "100_5_64_9.def" in f][0]
    model, _ = parse_file(file, max_horizon=1000)
    cp_model = CpSatMultiskillRcpspSolver(
        problem=model,
    )
    cp_model.init_model(
        one_worker_per_task=True,
    )
    cp_model.cp_model.Minimize(cp_model.variables["makespan"])
    p = ParametersCp.default_cpsat()
    res = cp_model.solve(parameters_cp=p, time_limit=20)
    solution = res.get_best_solution_fit()[0]
    assert model.satisfy(solution)


def test_imopse_cpsat_with_calendar():
    """
    To test some part of the cp model relative to calendar handling
    """
    file = [f for f in get_data_available() if "100_5_64_9.def" in f][0]
    model, _ = parse_file(file, max_horizon=1000)
    for emp in model.employees:
        model.employees[emp].calendar_employee = np.array(
            model.employees[emp].calendar_employee
        )
        model.employees[emp].calendar_employee[5:10] = 0
    model.update_functions()
    cp_model = CpSatMultiskillRcpspSolver(
        problem=model,
    )
    cp_model.init_model(
        one_worker_per_task=True,
    )
    cp_model.cp_model.Minimize(cp_model.variables["makespan"])
    p = ParametersCp.default_cpsat()
    res = cp_model.solve(parameters_cp=p, time_limit=20)
    solution = res.get_best_solution_fit()[0]
    assert model.satisfy(solution)


@pytest.fixture()
def problem():
    file = [f for f in get_data_available() if "100_5_64_9.def" in f][0]
    problem, _ = parse_file(file, max_horizon=1000)
    for emp in problem.employees:
        problem.employees[emp].calendar_employee = np.array(
            problem.employees[emp].calendar_employee
        )
        problem.employees[emp].calendar_employee[5:10] = 0
    problem.update_functions()
    return problem


@pytest.fixture()
def solver(problem):
    return CpSatMultiskillRcpspSolver(
        problem=problem,
    )


@pytest.fixture()
def problem_multimode(problem):
    task = 2
    old_mode = 1
    new_mode = 2
    new_details = dict(problem.mode_details[task][old_mode])
    new_details["duration"] *= 10
    new_details["Q8"] = 2
    problem.mode_details[task][new_mode] = new_details
    problem.update_functions()
    return problem


@pytest.fixture
def random_seed():
    random.seed(0)
    np.random.seed(0)


@pytest.mark.parametrize(
    "task, start_or_end, sign , time",
    [
        (2, StartOrEnd.START, SignEnum.UEQ, 600),
    ],
)
def test_task_constraint(problem, solver, task, start_or_end, sign, time):
    sol: MultiskillRcpspSolution = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()

    # before adding the constraint, not already satisfied
    assert not sol.constraint_on_task_satisfied(
        task=task, start_or_end=start_or_end, sign=sign, time=time
    )
    # add constraint: should be now satisfied
    cstrs = solver.add_constraint_on_task(
        task=task, start_or_end=start_or_end, sign=sign, time=time
    )
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert sol.constraint_on_task_satisfied(
        task=task, start_or_end=start_or_end, sign=sign, time=time
    )
    # check constraints can be effectively removed
    solver.remove_constraints(cstrs)
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert not sol.constraint_on_task_satisfied(
        task=task, start_or_end=start_or_end, sign=sign, time=time
    )


def test_constraint_nb_allocation_changes(problem):
    solver = CpSatMultiskillRcpspSolver(
        problem=problem,
    )
    res = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=5)])
    print(len(res))
    ref: MultiskillRcpspSolution = res.get_best_solution()
    sol: MultiskillRcpspSolution
    for sol, _ in res:
        print(sol.compute_nb_allocation_changes(ref))
    sol, _ = res[0]
    nb_changes_max = 4
    assert sol.compute_nb_allocation_changes(ref) > nb_changes_max
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert sol.compute_nb_allocation_changes(ref) > nb_changes_max
    solver.add_constraint_on_nb_allocation_changes(ref=ref, nb_changes=nb_changes_max)
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert sol.compute_nb_allocation_changes(ref) <= nb_changes_max


def test_constraint_same_allocation_as_ref(problem):
    solver = CpSatMultiskillRcpspSolver(
        problem=problem,
    )
    res = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=5)])
    print(len(res))
    ref: MultiskillRcpspSolution = res.get_best_solution()
    sol: MultiskillRcpspSolution
    for sol, _ in res:
        print(sol.compute_nb_allocation_changes(ref))
    nb_changes_max = 4
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert sol.compute_nb_allocation_changes(ref) > nb_changes_max
    constraints = solver.add_constraint_same_allocation_as_ref(ref=ref)
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert sol.check_same_allocation_as_ref(ref)

    task = 101
    employee = 3
    subtasks = [t for t in problem.tasks_list if t != task]
    subresources = [e for e in problem.unary_resources_list if e != employee]

    solver.remove_constraints(constraints)
    solver.add_constraint_same_allocation_as_ref(
        ref, tasks=subtasks, unary_resources=subresources
    )
    solver.add_constraint_on_task_unary_resource_allocation(
        task=task, unary_resource=employee, used=True
    )
    solver.add_constraint_on_task_unary_resource_allocation(
        task=98, unary_resource=employee, used=True
    )
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert sol.check_same_allocation_as_ref(
        ref, tasks=subtasks, unary_resources=subresources
    )
    assert not sol.check_same_allocation_as_ref(ref)


def test_constraint_nb_usages(problem):
    solver = CpSatMultiskillRcpspSolver(
        problem=problem,
    )
    sol: MultiskillRcpspSolution = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    nb_usages_total = sol.compute_nb_unary_resource_usages()
    solver.add_constraint_on_total_nb_usages(sign=SignEnum.UP, target=nb_usages_total)
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert sol.compute_nb_unary_resource_usages() > nb_usages_total

    unary_resource = 3
    nb_usages = sol.compute_nb_unary_resource_usages(unary_resources=(unary_resource,))
    solver.add_constraint_on_unary_resource_nb_usages(
        sign=SignEnum.UP, target=nb_usages, unary_resource=unary_resource
    )
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert (
        sol.compute_nb_unary_resource_usages(unary_resources=(unary_resource,))
        > nb_usages
    )


def test_objective_global_makespan(problem):
    solver = CpSatMultiskillRcpspSolver(
        problem=problem,
    )
    solver.init_model()

    objective = solver.get_global_makespan_variable()
    solver.minimize_variable(objective)
    sol: MultiskillRcpspSolution
    sol, _ = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])[-1]
    assert solver.solver.ObjectiveValue() == sol.get_max_end_time()


def test_objective_subtasks_makespan(problem):
    solver = CpSatMultiskillRcpspSolver(
        problem=problem,
    )
    solver.init_model()
    subtasks = [2, 3]

    objective = solver.get_subtasks_makespan_variable(subtasks)
    solver.minimize_variable(objective)
    sol: MultiskillRcpspSolution
    sol, _ = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])[-1]
    assert solver.solver.ObjectiveValue() == max(
        sol.get_end_time(task) for task in subtasks
    )


def test_objective_subtasks_sum_starts(problem):
    solver = CpSatMultiskillRcpspSolver(
        problem=problem,
    )
    solver.init_model()
    subtasks = [2, 3]
    objective = solver.get_subtasks_sum_start_time_variable(subtasks)
    solver.minimize_variable(objective)
    sol: MultiskillRcpspSolution
    sol, _ = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])[-1]
    assert solver.solver.ObjectiveValue() == sum(
        sol.get_start_time(task) for task in subtasks
    )


def test_objective_subtasks_sum_ends(problem):
    solver = CpSatMultiskillRcpspSolver(
        problem=problem,
    )
    solver.init_model()
    subtasks = [2, 3]
    objective = solver.get_subtasks_sum_end_time_variable(subtasks)
    solver.minimize_variable(objective)
    sol: MultiskillRcpspSolution
    sol, _ = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])[-1]
    assert solver.solver.ObjectiveValue() == sum(
        sol.get_end_time(task) for task in subtasks
    )


def test_objective_nb_tasks_done(problem):
    solver = CpSatMultiskillRcpspSolver(
        problem=problem,
    )
    solver.init_model()
    objective = -solver.get_nb_tasks_done_variable()
    solver.minimize_variable(objective)
    sol: MultiskillRcpspSolution
    sol, _ = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])[-1]
    assert solver.solver.ObjectiveValue() == -sum(
        max(
            sol.is_allocated(task, unary_resource)
            for unary_resource in problem.unary_resources_list
        )
        for task in problem.tasks_list
    )


def test_objective_nb_unary_resources_used(problem):
    solver = CpSatMultiskillRcpspSolver(
        problem=problem,
    )
    solver.init_model()
    objective = solver.get_nb_unary_resources_used_variable()
    solver.minimize_variable(objective)
    sol: MultiskillRcpspSolution
    sol, _ = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])[-1]
    assert solver.solver.ObjectiveValue() == sum(
        max(sol.is_allocated(task, unary_resource) for task in problem.tasks_list)
        for unary_resource in problem.unary_resources_list
    )


def test_constraint_multimode(problem_multimode, random_seed):
    problem = problem_multimode
    assert problem.is_multimode
    solver = CpSatMultiskillRcpspSolver(
        problem=problem,
    )
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        parameters_cp=ParametersCp.default(),
    ).get_best_solution()
    task = 2
    mode = 2
    assert not sol.get_mode(task) == mode

    solver.add_constraint_on_task_mode(task, mode)
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert sol.get_mode(task) == mode

    task_nok = 3
    mode_nok = 2
    with pytest.raises(ValueError):
        solver.add_constraint_on_task_mode(task_nok, mode_nok)
