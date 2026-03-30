#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

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
from discrete_optimization.rcpsp.problem import RcpspProblem
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
from discrete_optimization.rcpsp.special_constraints import (
    SpecialConstraintsDescription,
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
    solution: RcpspSolution
    solution, fit = result_storage.get_best_solution_fit()
    solution_rebuilt = RcpspSolution(
        problem=rcpsp_problem,
        rcpsp_permutation=solution.rcpsp_permutation,
        rcpsp_modes=solution.rcpsp_modes,
    )
    fit_2 = rcpsp_problem.evaluate(solution_rebuilt)
    assert fit == -fit_2["makespan"]
    assert rcpsp_problem.satisfy(solution)
    assert solution.check_all_resource_capacity_constraints()
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
    solver.init_model()

    task = 2
    print(problem.mode_details[task])

    mode = 1
    constraints = solver.add_constraint_on_task_mode(task, mode)
    sol = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)]
    ).get_best_solution()
    assert sol.get_mode(task) == mode

    solver.remove_constraints(constraints)
    mode = 3
    constraints = solver.add_constraint_on_task_mode(task, mode)
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
    assert solution.check_all_resource_capacity_constraints()

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
    assert solution.check_all_resource_capacity_constraints()


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
    assert solution.check_all_resource_capacity_constraints()


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


def test_special_constraints():
    """Test that special constraints are correctly enforced by CP-SAT solver."""
    mode_details = {
        1: {1: {"duration": 0}},  # dummy start
        2: {1: {"duration": 3, "R1": 1}},
        3: {1: {"duration": 2, "R1": 1}},
        4: {1: {"duration": 4, "R1": 1}},
        5: {1: {"duration": 0}},  # dummy end
    }

    successors = {
        1: [2, 3],
        2: [5],
        3: [4],
        4: [5],
        5: [],
    }

    resources = {"R1": 2}

    # Test 1: start_together constraint
    special_constraints_1 = SpecialConstraintsDescription(
        start_together=[(2, 3)],  # tasks 2 and 3 start together
    )

    problem_1 = RcpspProblem(
        resources=resources,
        non_renewable_resources=[],
        mode_details=mode_details,
        successors=successors,
        horizon=100,
        special_constraints=special_constraints_1,
    )

    solver_1 = CpSatRcpspSolver(problem=problem_1)
    result_1 = solver_1.solve(time_limit=10)
    solution_1 = result_1.get_best_solution()

    assert solution_1 is not None, "Solver should find a solution"
    assert solution_1.get_start_time(2) == solution_1.get_start_time(3), (
        f"start_together constraint not satisfied: "
        f"task 2 starts at {solution_1.get_start_time(2)}, "
        f"task 3 starts at {solution_1.get_start_time(3)}"
    )
    assert problem_1.satisfy(solution_1), "Solution should satisfy all constraints"

    # Test 2: start_at_end constraint
    special_constraints_2 = SpecialConstraintsDescription(
        start_at_end=[(3, 4)],  # task 4 starts when task 3 ends
    )

    problem_2 = RcpspProblem(
        resources=resources,
        non_renewable_resources=[],
        mode_details=mode_details,
        successors=successors,
        horizon=100,
        special_constraints=special_constraints_2,
    )

    solver_2 = CpSatRcpspSolver(problem=problem_2)
    result_2 = solver_2.solve(time_limit=10)
    solution_2 = result_2.get_best_solution()

    assert solution_2 is not None, "Solver should find a solution"
    assert solution_2.get_end_time(3) == solution_2.get_start_time(4), (
        f"start_at_end constraint not satisfied: "
        f"task 3 ends at {solution_2.get_end_time(3)}, "
        f"task 4 starts at {solution_2.get_start_time(4)}"
    )
    assert problem_2.satisfy(solution_2), "Solution should satisfy all constraints"

    # Test 3: start_times_window constraint
    special_constraints_3 = SpecialConstraintsDescription(
        start_times_window={2: (5, 10)},  # task 2 must start between 5 and 10
    )

    problem_3 = RcpspProblem(
        resources=resources,
        non_renewable_resources=[],
        mode_details=mode_details,
        successors=successors,
        horizon=100,
        special_constraints=special_constraints_3,
    )

    solver_3 = CpSatRcpspSolver(problem=problem_3)
    result_3 = solver_3.solve(time_limit=10)
    solution_3 = result_3.get_best_solution()

    assert solution_3 is not None, "Solver should find a solution"
    assert 5 <= solution_3.get_start_time(2) <= 10, (
        f"start_times_window constraint not satisfied: "
        f"task 2 starts at {solution_3.get_start_time(2)}"
    )
    assert problem_3.satisfy(solution_3), "Solution should satisfy all constraints"

    # Test 4: disjunctive_tasks constraint (with parallel successors)
    successors_4 = {
        1: [2, 3],
        2: [5],
        3: [5],
        4: [5],
        5: [],
    }  # tasks 2 and 3 are parallel
    special_constraints_4 = SpecialConstraintsDescription(
        disjunctive_tasks=[(2, 3)],  # tasks 2 and 3 cannot overlap
    )

    problem_4 = RcpspProblem(
        resources=resources,
        non_renewable_resources=[],
        mode_details=mode_details,
        successors=successors_4,
        horizon=100,
        special_constraints=special_constraints_4,
    )

    solver_4 = CpSatRcpspSolver(problem=problem_4)
    result_4 = solver_4.solve(time_limit=10)
    solution_4 = result_4.get_best_solution()

    assert solution_4 is not None, "Solver should find a solution"
    task2_end = solution_4.get_end_time(2)
    task3_start = solution_4.get_start_time(3)
    task3_end = solution_4.get_end_time(3)
    task2_start = solution_4.get_start_time(2)
    no_overlap = (task2_end <= task3_start) or (task3_end <= task2_start)
    assert no_overlap, (
        f"disjunctive_tasks constraint not satisfied: "
        f"task 2 [{task2_start}, {task2_end}], "
        f"task 3 [{task3_start}, {task3_end}]"
    )
    assert problem_4.satisfy(solution_4), "Solution should satisfy all constraints"


def test_start_to_start_min_time_lag_negative_offset():
    """Test start_to_start_min_time_lag constraint with negative offsets in both CP-SAT and SGS."""
    mode_details = {
        1: {1: {"duration": 0}},  # dummy start
        2: {1: {"duration": 5, "R1": 1}},
        3: {1: {"duration": 3, "R1": 1}},
        4: {1: {"duration": 2, "R1": 1}},
        5: {1: {"duration": 0}},  # dummy end
    }

    successors = {
        1: [2, 3, 4],
        2: [5],
        3: [5],
        4: [5],
        5: [],
    }

    resources = {"R1": 2}

    # Test with negative offset: task 3 must start at least 3 time units BEFORE task 2
    # Constraint: start(2) + (-3) <= start(3) => start(2) - 3 <= start(3) => start(3) >= start(2) - 3
    special_constraints = SpecialConstraintsDescription(
        start_to_start_min_time_lag=[(2, 3, -3)],
    )

    problem = RcpspProblem(
        resources=resources,
        non_renewable_resources=[],
        mode_details=mode_details,
        successors=successors,
        horizon=100,
        special_constraints=special_constraints,
    )

    # Test CP-SAT solver
    solver = CpSatRcpspSolver(problem=problem)
    result = solver.solve(time_limit=10)
    solution = result.get_best_solution()

    assert solution is not None, "Solver should find a solution"

    # Verify the constraint: start(2) - 3 <= start(3)
    start_2 = solution.get_start_time(2)
    start_3 = solution.get_start_time(3)
    assert start_2 - 3 <= start_3, (
        f"start_to_start_min_time_lag constraint not satisfied: "
        f"start(2)={start_2}, start(3)={start_3}, "
        f"but start(2) - 3 = {start_2 - 3} > {start_3}"
    )
    assert problem.satisfy(solution), "Solution should satisfy all constraints"

    # Test SGS schedule generation
    # Create a solution with a specific permutation
    from discrete_optimization.rcpsp.solution import (
        generate_schedule_from_permutation_serial_sgs_special_constraints,
    )

    # Try different permutations to test both orderings
    for permutation in [[0, 1, 2], [1, 0, 2]]:  # task 2 before 3, and task 3 before 2
        test_solution = RcpspSolution(
            problem=problem,
            rcpsp_permutation=permutation,
            rcpsp_modes=[1, 1, 1],
        )

        schedule, feasible = (
            generate_schedule_from_permutation_serial_sgs_special_constraints(
                test_solution, problem
            )
        )

        assert feasible, (
            f"SGS should generate feasible schedule for permutation {permutation}"
        )

        # Verify the constraint in the generated schedule
        sgs_start_2 = schedule[2]["start_time"]
        sgs_start_3 = schedule[3]["start_time"]
        assert sgs_start_2 - 3 <= sgs_start_3, (
            f"SGS violated start_to_start_min_time_lag constraint with permutation {permutation}: "
            f"start(2)={sgs_start_2}, start(3)={sgs_start_3}, "
            f"but start(2) - 3 = {sgs_start_2 - 3} > {sgs_start_3}"
        )


def test_start_to_start_min_time_lag_positive_offset():
    """Test start_to_start_min_time_lag constraint with positive offset to ensure SGS precedence logic works."""
    mode_details = {
        1: {1: {"duration": 0}},  # dummy start
        2: {1: {"duration": 2, "R1": 1}},
        3: {1: {"duration": 3, "R1": 1}},
        4: {1: {"duration": 0}},  # dummy end
    }

    successors = {
        1: [2, 3],
        2: [4],
        3: [4],
        4: [],
    }

    resources = {"R1": 1}

    # Test with positive offset: task 3 must start at least 5 units after task 2 starts
    # Constraint: start(2) + 5 <= start(3)
    special_constraints = SpecialConstraintsDescription(
        start_to_start_min_time_lag=[(2, 3, 5)],
    )

    problem = RcpspProblem(
        resources=resources,
        non_renewable_resources=[],
        mode_details=mode_details,
        successors=successors,
        horizon=100,
        special_constraints=special_constraints,
    )

    solver = CpSatRcpspSolver(problem=problem)
    result = solver.solve(time_limit=10)
    solution = result.get_best_solution()

    assert solution is not None, "Solver should find a solution"

    # Verify the constraint: start(2) + 5 <= start(3)
    start_2 = solution.get_start_time(2)
    start_3 = solution.get_start_time(3)
    assert start_2 + 5 <= start_3, (
        f"start_to_start_min_time_lag constraint not satisfied: "
        f"start(2)={start_2}, start(3)={start_3}, "
        f"but start(2) + 5 = {start_2 + 5} > {start_3}"
    )
    assert problem.satisfy(solution), "Solution should satisfy all constraints"
