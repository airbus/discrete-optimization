#  Copyright (c) 2024-2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from pytest_cases import fixture

from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.cp_tools import ParametersCp, SignEnum
from discrete_optimization.generic_tools.lns_cp import LnsOrtoolsCpSat
from discrete_optimization.generic_tools.lns_tools import ConstraintHandlerMix
from discrete_optimization.tsp.problem import TspSolution
from discrete_optimization.tsp.solvers.cpsat import CpSatTspSolver
from discrete_optimization.tsp.solvers.lns_cpsat import (
    SubpathTspConstraintHandler,
    TspConstraintHandler,
)


@fixture
def solver(problem):
    solver = CpSatTspSolver(problem)
    solver.init_model()
    return solver


def test_cpsat_solver(problem, solver, end_index, start_solutions):
    sol1, sol2 = start_solutions

    solver.set_warm_start(sol1)
    res = solver.solve(
        callbacks=[
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            ),
            NbIterationStopper(nb_iteration_max=1),
        ],
        ortools_cpsat_solver_kwargs=dict(fix_variables_to_their_hinted_value=True),
        parameters_cp=ParametersCp.default_cpsat(),
    )
    sol, fitness = res.get_best_solution_fit()
    sol: TspSolution
    assert problem.satisfy(sol)
    assert sol.end_index == end_index
    assert sol.permutation == sol1.permutation

    # warm start at first solution
    solver.set_warm_start(sol2)
    # force first solution to be the hinted one
    res = solver.solve(
        parameters_cp=ParametersCp.default_cpsat(),
        ortools_cpsat_solver_kwargs=dict(fix_variables_to_their_hinted_value=True),
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    )
    assert res[0][0].permutation == sol2.permutation

    # test get_start_time, get_end_time
    for task in problem.tasks_list:
        sol.get_start_time(task), sol.get_end_time(task)


def test_subobjectives(problem, solver):
    objective = solver.get_global_makespan_variable()
    solver.minimize_variable(objective)
    sol, _ = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])[-1]
    solver.solver.ObjectiveValue() == sol.get_max_end_time()

    subtasks = {1, 3}
    # max end time subtasks
    objective = solver.get_subtasks_makespan_variable(subtasks)
    solver.minimize_variable(objective)
    sol, _ = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])[-1]
    solver.solver.ObjectiveValue() == max(sol.get_end_time(task) for task in subtasks)
    # sum end time subtasks
    objective = solver.get_subtasks_sum_end_time_variable(subtasks)
    solver.minimize_variable(objective)
    sol, _ = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])[-1]
    solver.solver.ObjectiveValue() == sum(sol.get_end_time(task) for task in subtasks)
    # sum start time subtasks
    objective = solver.get_subtasks_sum_start_time_variable(subtasks)
    solver.minimize_variable(objective)
    sol, _ = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])[-1]
    solver.solver.ObjectiveValue() == sum(sol.get_start_time(task) for task in subtasks)
    # max end time
    objective = solver.get_global_makespan_variable()
    solver.minimize_variable(objective)
    sol, _ = solver.solve(callbacks=[NbIterationStopper(nb_iteration_max=1)])[-1]
    solver.solver.ObjectiveValue() == sol.get_max_end_time()


def test_task_constraint(problem, solver):
    task = 2
    start_or_end = StartOrEnd.END
    sign = SignEnum.LEQ
    antisign = SignEnum.UP
    time = 2

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


def test_chaining_tasks_constraint(problem, solver, start_solutions):
    # add warm start to ensure tasks not chained at first solve
    start_solution = start_solutions[0]
    task1 = start_solution.permutation[0]
    task2 = start_solution.permutation[2]

    # before adding the constraint, not already satisfied
    solver.set_warm_start(start_solution)
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
    ).get_best_solution()
    assert not sol.constraint_chaining_tasks_satisfied(task1=task1, task2=task2)
    # add constraint
    cstrs = solver.add_constraint_chaining_tasks(task1=task1, task2=task2)
    solver.set_warm_start(start_solution)
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=2)],
    ).get_best_solution()
    assert sol.constraint_chaining_tasks_satisfied(task1=task1, task2=task2)
    # remove constraint
    solver.remove_constraints(cstrs)
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert not sol.constraint_chaining_tasks_satisfied(task1=task1, task2=task2)


def test_lns_cpsat_solver(problem, solver):
    p = ParametersCp.default_cpsat()
    lns_solver = LnsOrtoolsCpSat(
        problem=problem,
        subsolver=solver,
        constraint_handler=ConstraintHandlerMix(
            problem=problem,
            list_constraints_handler=[
                SubpathTspConstraintHandler(
                    problem=problem, fraction_segment_to_fix=0.7
                ),
                TspConstraintHandler(problem=problem, fraction_segment_to_fix=0.7),
            ],
            list_proba=[0.5, 0.5],
        ),
    )
    res = lns_solver.solve(
        skip_initial_solution_provider=True,
        nb_iteration_lns=2,
        time_limit_subsolver=2,
        callbacks=[
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
        parameters_cp=p,
    )
    sol, fitness = res.get_best_solution_fit()
    assert problem.satisfy(sol)
