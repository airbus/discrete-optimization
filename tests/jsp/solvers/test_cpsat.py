#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import random

import numpy as np
import pytest

from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp, SignEnum
from discrete_optimization.jsp.parser import get_data_available, parse_file
from discrete_optimization.jsp.problem import JobShopProblem, JobShopSolution
from discrete_optimization.jsp.solvers.cpsat import CpSatJspSolver


@pytest.fixture
def random_seed():
    random.seed(0)
    np.random.seed(0)


@pytest.fixture()
def problem() -> JobShopProblem:
    filename = "la02"
    filepath = [f for f in get_data_available() if f.endswith(filename)][0]
    return parse_file(filepath)


def test_cpsat_jsp(random_seed, problem):
    solver = CpSatJspSolver(problem=problem)
    parameters_cp = ParametersCp.default()
    sol: JobShopSolution = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        parameters_cp=parameters_cp,
    ).get_best_solution()
    print(sol.schedule)
    assert problem.satisfy(sol)

    # test prob.task_lists and sol.get_start_stime/end_time
    assert len(problem.tasks_list) == problem.n_all_jobs
    for task in problem.tasks_list:
        print(sol.get_start_time(task), sol.get_end_time(task))

    # test subojectives

    # max end time subtasks
    subtasks = {(0, 1), (1, 2)}
    objective = solver.get_subtasks_makespan_variable(subtasks)
    solver.minimize_variable(objective)
    sol, _ = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])[-1]
    solver.solver.ObjectiveValue() == max(sol.get_end_time(task) for task in subtasks)
    # sum end time subtasks
    subtasks = {(0, 1), (1, 2)}
    objective = solver.get_subtasks_sum_end_time_variable(subtasks)
    solver.minimize_variable(objective)
    sol, _ = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])[-1]
    solver.solver.ObjectiveValue() == sum(sol.get_end_time(task) for task in subtasks)
    # sum start time subtasks
    subtasks = {(0, 1), (1, 2)}
    objective = solver.get_subtasks_sum_start_time_variable(subtasks)
    solver.minimize_variable(objective)
    sol, _ = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])[-1]
    solver.solver.ObjectiveValue() == sum(sol.get_start_time(task) for task in subtasks)
    # max end time
    objective = solver.get_global_makespan_variable()
    solver.minimize_variable(objective)
    sol, _ = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])[-1]
    solver.solver.ObjectiveValue() == sol.get_max_end_time()


@pytest.mark.parametrize(
    "task, start_or_end, sign , time",
    [
        ((0, 1), StartOrEnd.START, SignEnum.UEQ, 120),
        ((0, 1), StartOrEnd.START, SignEnum.UP, 120),
        ((0, 2), StartOrEnd.START, SignEnum.LEQ, 120),
        ((0, 2), StartOrEnd.START, SignEnum.LESS, 120),
        ((0, 2), StartOrEnd.START, SignEnum.EQUAL, 120),
        ((0, 0), StartOrEnd.END, SignEnum.UEQ, 120),
    ],
)
def test_task_constraint(problem, task, start_or_end, sign, time, random_seed):
    solver = CpSatJspSolver(problem=problem)
    parameters_cp = ParametersCp.default()
    sol: JobShopSolution = solver.solve(
        parameters_cp=parameters_cp, callbacks=[NbIterationStopper(nb_iteration_max=1)]
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
        parameters_cp=parameters_cp,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        time_limit=1000,
    ).get_best_solution()
    assert sol.constraint_on_task_satisfied(
        task=task, start_or_end=start_or_end, sign=sign, time=time
    )
    # check constraints can be effectively removed
    solver.remove_constraints(cstrs)
    sol = solver.solve(
        parameters_cp=parameters_cp, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert not sol.constraint_on_task_satisfied(
        task=task, start_or_end=start_or_end, sign=sign, time=time
    )


def test_chaining_tasks_constraint(problem, random_seed):
    solver = CpSatJspSolver(problem=problem)
    parameters_cp = ParametersCp.default()
    task1 = (0, 0)
    task2 = (1, 4)

    # before adding the constraint, not already satisfied
    sol: JobShopSolution = solver.solve(
        parameters_cp=parameters_cp, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert not sol.constraint_chaining_tasks_satisfied(task1=task1, task2=task2)
    # add constraint
    cstrs = solver.add_constraint_chaining_tasks(task1=task1, task2=task2)
    sol = solver.solve(
        parameters_cp=parameters_cp, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert sol.constraint_chaining_tasks_satisfied(task1=task1, task2=task2)
    # remove constraint
    solver.remove_constraints(cstrs)
    sol = solver.solve(
        parameters_cp=parameters_cp, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert not sol.constraint_chaining_tasks_satisfied(task1=task1, task2=task2)
