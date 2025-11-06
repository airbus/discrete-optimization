#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import pytest

from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp, SignEnum
from discrete_optimization.singlemachine.problem import WTSolution
from discrete_optimization.singlemachine.solvers.cpsat import CpsatWTSolver


def test_cpsat(problem):
    solver = CpsatWTSolver(problem)
    res = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])
    sol = res.get_best_solution()
    assert problem.satisfy(sol)

    # warm-start
    ref = problem.get_dummy_solution()
    assert not sol.schedule == ref.schedule  # different sols before warmstart
    solver.set_warm_start(ref)
    res = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        ortools_cpsat_solver_kwargs=dict(
            log_search_progress=True, fix_variables_to_their_hinted_value=True
        ),
    )
    sol2, _ = res[0]

    assert sol2.schedule == ref.schedule  # same sols after warmstart

    # test prob.task_lists and sol.get_start_stime/end_time
    assert len(problem.tasks_list) == problem.num_jobs
    for task in problem.tasks_list:
        print(sol.get_start_time(task), sol.get_end_time(task))

    # test subojectives

    # max end time subtasks
    subtasks = {1, 4}
    objective = solver.get_subtasks_makespan_variable(subtasks)
    solver.minimize_variable(objective)
    sol, _ = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])[-1]
    solver.solver.ObjectiveValue() == max(sol.get_end_time(task) for task in subtasks)
    # sum end time subtasks
    subtasks = {1, 4}
    objective = solver.get_subtasks_sum_end_time_variable(subtasks)
    solver.minimize_variable(objective)
    sol, _ = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])[-1]
    solver.solver.ObjectiveValue() == sum(sol.get_end_time(task) for task in subtasks)
    # sum start time subtasks
    subtasks = {1, 4}
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
    "task, start_or_end, sign , time, antisign",
    [
        (0, StartOrEnd.START, SignEnum.UEQ, 1970, SignEnum.LESS),
        (1, StartOrEnd.END, SignEnum.LESS, 1990, SignEnum.UEQ),
    ],
)
def test_task_constraint(problem, task, start_or_end, sign, time, antisign):
    solver = CpsatWTSolver(problem)
    parameters_cp = ParametersCp.default()
    sol: WTSolution = solver.solve(
        parameters_cp=parameters_cp, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()

    # add anti-constraint
    cstrs = solver.add_constraint_on_task(
        task=task, start_or_end=start_or_end, sign=antisign, time=time
    )
    sol = solver.solve(
        parameters_cp=parameters_cp,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    ).get_best_solution()
    assert sol.constraint_on_task_satisfied(
        task=task, start_or_end=start_or_end, sign=antisign, time=time
    )

    # remove constraint and add real constraint
    solver.remove_constraints(cstrs)
    cstrs = solver.add_constraint_on_task(
        task=task, start_or_end=start_or_end, sign=sign, time=time
    )
    sol = solver.solve(
        parameters_cp=parameters_cp,
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    ).get_best_solution()
    assert sol.constraint_on_task_satisfied(
        task=task, start_or_end=start_or_end, sign=sign, time=time
    )


def test_chaining_tasks_constraint(problem):
    solver = CpsatWTSolver(problem=problem)
    parameters_cp = ParametersCp.default()
    task1 = 1
    task2 = 0

    # before adding the constraint, not already satisfied
    sol: WTSolution = solver.solve(
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
