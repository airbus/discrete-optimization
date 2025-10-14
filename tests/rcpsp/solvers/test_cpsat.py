import logging
import random

import numpy as np
import pytest

from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import SignEnum
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solution import RcpspSolution
from discrete_optimization.rcpsp.solvers.cpsat import (
    CpSatCumulativeResourceRcpspSolver,
    CpSatRcpspSolver,
    CpSatResourceRcpspSolver,
)
from discrete_optimization.rcpsp.solvers.pile import (
    PileCalendarRcpspSolver,
    PileRcpspSolver,
)
from discrete_optimization.rcpsp.utils import plot_task_gantt


@pytest.fixture
def random_seed():
    random.seed(0)
    np.random.seed(0)


@pytest.mark.parametrize(
    "model",
    ["j301_1.sm", "j1010_1.mm"],
)
def test_ortools(model):
    files_available = get_data_available()
    file = [f for f in files_available if model in f][0]
    rcpsp_problem = parse_file(file)
    solver = CpSatRcpspSolver(problem=rcpsp_problem)
    result_storage = solver.solve(time_limit=100)
    solution, fit = result_storage.get_best_solution_fit()
    solution_rebuilt = RcpspSolution(
        problem=rcpsp_problem,
        rcpsp_permutation=solution.rcpsp_permutation,
        rcpsp_modes=solution.rcpsp_modes,
    )
    fit_2 = rcpsp_problem.evaluate(solution_rebuilt)
    assert fit == -fit_2["makespan"]
    assert rcpsp_problem.satisfy(solution)
    rcpsp_problem.plot_ressource_view(solution)
    plot_task_gantt(rcpsp_problem, solution)

    # test warm start
    start_solution = (
        PileRcpspSolver(problem=rcpsp_problem).solve().get_best_solution_fit()[0]
    )

    # first solution is not start_solution
    assert result_storage[0][0].rcpsp_schedule != start_solution.rcpsp_schedule

    # warm start at first solution
    solver.set_warm_start(start_solution)
    # force first solution to be the hinted one
    result_storage = solver.solve(
        time_limit=100,
        ortools_cpsat_solver_kwargs=dict(fix_variables_to_their_hinted_value=True),
    )
    assert result_storage[0][0].rcpsp_schedule == start_solution.rcpsp_schedule


@pytest.mark.parametrize(
    "model",
    ["j301_1.sm", "j1010_1.mm"],
)
def test_objectives(model):
    files_available = get_data_available()
    file = [f for f in files_available if model in f][0]
    rcpsp_problem = parse_file(file)
    solver = CpSatRcpspSolver(problem=rcpsp_problem)
    solver.init_model()

    subtasks = {1, 4}
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


def test_mode_constraint_monomode():
    model = "j301_1.sm"
    files_available = get_data_available()
    file = [f for f in files_available if model in f][0]
    problem = parse_file(file)
    assert not problem.is_multimode

    solver = CpSatRcpspSolver(problem=problem)

    task = 2
    mode = 1
    solver.init_model()
    solver.add_constraint_on_task_mode(task, mode)
    sol: RcpspSolution = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert sol.get_mode(task) == mode

    with pytest.raises(ValueError):
        solver.add_constraint_on_task_mode(task, 0)


def test_mode_constraint_multimode(random_seed):
    model = "j1010_1.mm"
    files_available = get_data_available()
    file = [f for f in files_available if model in f][0]
    problem = parse_file(file)
    assert problem.is_multimode

    solver = CpSatRcpspSolver(problem=problem)

    task = 2
    print(problem.mode_details[task])
    mode = 1
    for _ in range(10):
        # at least 1 of 10 solution has a different mode
        sol: RcpspSolution = solver.solve(
            callbacks=[NbIterationStopper(nb_iteration_max=1)]
        ).get_best_solution()
        if sol.get_mode(task) != mode:
            break

    assert not sol.get_mode(task) == mode
    solver.add_constraint_on_task_mode(task, mode)
    for _ in range(3):
        # all 3 solutions have the fixed mode
        sol = solver.solve(
            callbacks=[NbIterationStopper(nb_iteration_max=1)]
        ).get_best_solution()
        assert sol.get_mode(task) == mode

    mode_nok = 15
    with pytest.raises(ValueError):
        solver.add_constraint_on_task_mode(task, mode_nok)


@pytest.mark.parametrize(
    "task, start_or_end, sign , time",
    [
        (2, StartOrEnd.START, SignEnum.UEQ, 6),
    ],
)
def test_task_constraint(task, start_or_end, sign, time):
    model = "j301_1.sm"
    files_available = get_data_available()
    file = [f for f in files_available if model in f][0]
    problem = parse_file(file)
    solver = CpSatRcpspSolver(problem=problem)
    sol: RcpspSolution = solver.solve(
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


def test_chaining_constraints():
    task1 = 4
    task2 = 2
    model = "j301_1.sm"
    files_available = get_data_available()
    file = [f for f in files_available if model in f][0]
    rcpsp_problem = parse_file(file)
    solver = CpSatRcpspSolver(problem=rcpsp_problem)
    sol: RcpspSolution = solver.solve(
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


@pytest.mark.parametrize(
    "model",
    ["j301_1.sm", "j1010_1.mm"],
)
def test_ortools_with_calendar_resource(model):
    files_available = get_data_available()
    file = [f for f in files_available if model in f][0]
    rcpsp_problem = parse_file(file)
    for resource in rcpsp_problem.resources:
        rcpsp_problem.resources[resource] = np.array(
            rcpsp_problem.get_resource_availability_array(resource)
        )
        rcpsp_problem.resources[resource][10:15] = 0
    rcpsp_problem.is_calendar = True
    rcpsp_problem.update_functions()
    solver = CpSatRcpspSolver(problem=rcpsp_problem)
    result_storage = solver.solve(time_limit=100)
    solution, fit = result_storage.get_best_solution_fit()
    solution_rebuilt = RcpspSolution(
        problem=rcpsp_problem,
        rcpsp_permutation=solution.rcpsp_permutation,
        rcpsp_modes=solution.rcpsp_modes,
    )
    fit_2 = rcpsp_problem.evaluate(solution_rebuilt)
    assert fit == -fit_2["makespan"]
    assert rcpsp_problem.satisfy(solution)
    rcpsp_problem.plot_ressource_view(solution)
    plot_task_gantt(rcpsp_problem, solution)

    # test warm start
    start_solution = (
        PileCalendarRcpspSolver(problem=rcpsp_problem)
        .solve()
        .get_best_solution_fit()[0]
    )

    # first solution is not start_solution
    assert result_storage[0][0].rcpsp_schedule != start_solution.rcpsp_schedule

    # warm start at first solution
    solver.set_warm_start(start_solution)
    # force first solution to be the hinted one
    result_storage = solver.solve(
        time_limit=100,
        ortools_cpsat_solver_kwargs=dict(fix_variables_to_their_hinted_value=True),
    )
    assert result_storage[0][0].rcpsp_schedule == start_solution.rcpsp_schedule


@pytest.mark.parametrize(
    "model",
    ["j301_1.sm", "j1010_1.mm"],
)
def test_ortools_cumulativeresource_optim(model):
    files_available = get_data_available()
    file = [f for f in files_available if model in f][0]
    rcpsp_problem = parse_file(file)
    solver = CpSatCumulativeResourceRcpspSolver(problem=rcpsp_problem)
    result_storage = solver.solve(time_limit=50)
    solution, fit = result_storage.get_best_solution_fit()
    assert rcpsp_problem.satisfy(solution)


@pytest.mark.parametrize(
    "model",
    ["j301_1.sm", "j1010_1.mm"],
)
def test_ortools_resource_optim(model):
    files_available = get_data_available()
    file = [f for f in files_available if model in f][0]
    rcpsp_problem = parse_file(file)
    solver = CpSatResourceRcpspSolver(problem=rcpsp_problem)
    result_storage = solver.solve(time_limit=50)
    solution, fit = result_storage.get_best_solution_fit()
    assert rcpsp_problem.satisfy(solution)


def test_ortools_with_cb(caplog, random_seed):
    model = "j1201_1.sm"
    files_available = get_data_available()
    file = [f for f in files_available if model in f][0]
    rcpsp_problem = parse_file(file)
    solver = CpSatRcpspSolver(problem=rcpsp_problem)

    class VariablePrinterCallback(Callback):
        def __init__(self) -> None:
            super().__init__()
            self.nb_solution = 0

        def on_step_end(self, step: int, res: ResultStorage, solver: CpSatRcpspSolver):
            self.nb_solution += 1
            sol: RcpspSolution
            sol, fit = res.list_solution_fits[-1]
            logging.debug(f"Solution #{self.nb_solution}:")
            logging.debug(sol.rcpsp_schedule)
            logging.debug(sol.rcpsp_modes)

    callbacks = [VariablePrinterCallback(), NbIterationStopper(1)]

    with caplog.at_level(logging.DEBUG):
        result_storage = solver.solve(callbacks=callbacks, time_limit=20)

    assert "Solution #1" in caplog.text
    assert (
        "stopped by user callback" in caplog.text
    )  # stopped by callback instead of ortools timer
    # only true if at least one solution found before 20s (ortools timer limit)
