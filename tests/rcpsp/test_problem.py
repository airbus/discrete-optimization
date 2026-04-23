#  Copyright (c) 2022-2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import random
from collections.abc import Hashable
from math import isclose

import pytest

from discrete_optimization.datasets import get_data_home
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp.solution import RcpspSolution, TaskDetails

files_rcpsp = get_data_available()
single_modes_files = [f for f in files_rcpsp if "sm" in f]

multi_modes_files = [f for f in files_rcpsp if "mm" in f]

data_folder_rcp = f"{get_data_home()}/rcpsp/RG30/Set 1/"
files_patterson = get_data_available(data_folder=data_folder_rcp)


@pytest.mark.parametrize("rcpsp_problem_file", single_modes_files + files_patterson)
def test_single_mode(rcpsp_problem_file):
    rcpsp_problem: RcpspProblem = parse_file(rcpsp_problem_file)
    assert rcpsp_problem.is_rcpsp_multimode() is False
    assert rcpsp_problem.is_varying_resource() is False
    # Create solution (mode = 1 for each task, identity permutation)
    permutation = [i for i in range(rcpsp_problem.n_jobs_non_dummy)]
    random.shuffle(permutation)
    mode_list = [1 for i in range(rcpsp_problem.n_jobs_non_dummy)]
    rcpsp_sol = RcpspSolution(
        problem=rcpsp_problem, rcpsp_permutation=permutation, rcpsp_modes=mode_list
    )
    evaluation = rcpsp_problem.evaluate(rcpsp_sol)
    assert rcpsp_problem.satisfy(rcpsp_sol)
    assert rcpsp_sol.rcpsp_schedule_feasible
    assert rcpsp_sol.rcpsp_modes == mode_list


def test_non_existing_modes_solution():
    files = get_data_available()
    files = [f for f in files if "j301_1.sm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_problem = parse_file(file_path)

    permutation = [i for i in range(rcpsp_problem.n_jobs_non_dummy)]
    mode_list = [2 for i in range(rcpsp_problem.n_jobs_non_dummy)]
    rcpsp_sol = RcpspSolution(
        problem=rcpsp_problem, rcpsp_permutation=permutation, rcpsp_modes=mode_list
    )
    assert not rcpsp_sol.rcpsp_schedule_feasible
    assert rcpsp_sol.rcpsp_modes == mode_list
    assert not rcpsp_problem.satisfy(rcpsp_sol)


def test_unfeasible_modes_solution():
    files = get_data_available()
    files = [f for f in files if "j1010_5.mm" in f]  # Multi-mode RCPSP
    file_path = files[0]
    rcpsp_problem = parse_file(file_path)
    permutation = [i for i in range(rcpsp_problem.n_jobs_non_dummy)]
    unfeasible_modes = [2, 1, 1, 1, 3, 1, 2, 1, 1, 3]
    rcpsp_sol = RcpspSolution(
        problem=rcpsp_problem,
        rcpsp_permutation=permutation,
        rcpsp_modes=unfeasible_modes,
    )
    assert not rcpsp_sol.rcpsp_schedule_feasible
    assert rcpsp_sol.rcpsp_modes == unfeasible_modes
    assert not rcpsp_problem.satisfy(rcpsp_sol)
    evaluation = rcpsp_problem.evaluate(rcpsp_sol)
    assert evaluation == {
        "makespan": 99999999,
        "mean_resource_reserve": 0,
        "constraint_penalty": 0.0,
    }


def test_feasible_modes_solution():
    files = get_data_available()
    files = [f for f in files if "j1010_1.mm" in f]  # Multi-mode RCPSP
    file_path = files[0]
    rcpsp_problem = parse_file(file_path)

    permutation = [i for i in range(rcpsp_problem.n_jobs_non_dummy)]
    feasible_modes = [1, 1, 3, 3, 2, 2, 3, 3, 3, 2]

    rcpsp_sol = RcpspSolution(
        problem=rcpsp_problem, rcpsp_permutation=permutation, rcpsp_modes=feasible_modes
    )
    assert rcpsp_sol.rcpsp_schedule_feasible
    assert rcpsp_sol.rcpsp_schedule == {
        1: {"start_time": 0, "end_time": 0},
        2: {"start_time": 0, "end_time": 1},
        3: {"start_time": 0, "end_time": 1},
        4: {"start_time": 1, "end_time": 11},
        5: {"start_time": 1, "end_time": 11},
        6: {"start_time": 11, "end_time": 14},
        7: {"start_time": 14, "end_time": 22},
        8: {"start_time": 14, "end_time": 21},
        9: {"start_time": 22, "end_time": 31},
        10: {"start_time": 22, "end_time": 27},
        11: {"start_time": 11, "end_time": 17},
        12: {"start_time": 31, "end_time": 31},
    }
    assert rcpsp_sol.rcpsp_modes == feasible_modes
    assert rcpsp_problem.satisfy(rcpsp_sol)
    assert isclose(rcpsp_sol.compute_mean_resource_reserve(), 0.5028790398735252)


def create_task_details_classic(
    solution: RcpspSolution, time_to_cut: int
) -> tuple[dict[Hashable, TaskDetails], dict[Hashable, TaskDetails]]:
    finished = set(
        [t for t in solution.rcpsp_schedule if solution.get_end_time(t) <= time_to_cut]
    )
    completed = {
        t: TaskDetails(solution.get_start_time(t), solution.get_end_time(t))
        for t in finished
    }
    ongoing = {
        t: TaskDetails(solution.get_start_time(t), solution.get_end_time(t))
        for t in solution.rcpsp_schedule
        if solution.get_start_time(t) <= time_to_cut < solution.get_end_time(t)
    }
    return completed, ongoing


@pytest.mark.parametrize("rcpsp_problem_file", single_modes_files)
def test_partial_sgs(rcpsp_problem_file):
    rcpsp_problem: RcpspProblem = parse_file(rcpsp_problem_file)
    assert rcpsp_problem.is_rcpsp_multimode() is False
    assert rcpsp_problem.is_varying_resource() is False
    # Create solution (mode = 1 for each task, identity permutation)
    permutation = [i for i in range(rcpsp_problem.n_jobs_non_dummy)]
    random.shuffle(permutation)
    mode_list = [1 for i in range(rcpsp_problem.n_jobs_non_dummy)]
    rcpsp_sol = RcpspSolution(
        problem=rcpsp_problem,
        rcpsp_permutation=permutation,
        rcpsp_modes=mode_list,
        fast=True,
    )
    time_to_cut = int(rcpsp_sol.get_end_time(rcpsp_problem.sink_task) / 2)
    completed, ongoing = create_task_details_classic(
        solution=rcpsp_sol, time_to_cut=time_to_cut
    )
    scheduled_start_time = {o: ongoing[o].start for o in ongoing}
    rcpsp_sol_copy = rcpsp_sol.copy()
    rcpsp_sol_copy.rcpsp_schedule[rcpsp_problem.sink_task]

    rcpsp_sol_copy.generate_schedule_from_permutation_serial_sgs_2(
        current_t=time_to_cut,
        completed_tasks=completed,
        scheduled_tasks_start_times=scheduled_start_time,
        do_fast=True,
    )
    for task in completed:
        assert completed[task].start == rcpsp_sol_copy.get_start_time(task)
        assert completed[task].end == rcpsp_sol_copy.get_end_time(task)
    for task in ongoing:
        assert ongoing[task].start == rcpsp_sol_copy.get_start_time(task)
        assert ongoing[task].end == rcpsp_sol_copy.get_end_time(task)

    rcpsp_sol_copy.rcpsp_schedule[rcpsp_problem.sink_task]
    rcpsp_sol_copy.generate_schedule_from_permutation_serial_sgs_2(
        current_t=time_to_cut,
        completed_tasks=completed,
        scheduled_tasks_start_times=scheduled_start_time,
        do_fast=False,
    )
    for task in completed:
        assert completed[task].start == rcpsp_sol_copy.get_start_time(task)
        assert completed[task].end == rcpsp_sol_copy.get_end_time(task)
    for task in ongoing:
        assert ongoing[task].start == rcpsp_sol_copy.get_start_time(task)
        assert ongoing[task].end == rcpsp_sol_copy.get_end_time(task)
    rcpsp_sol_copy.rcpsp_schedule[rcpsp_problem.sink_task]


@pytest.fixture
def small_problem():
    resources = {
        "R1": [0, 5, 5, 2, 3, 2, 2, 2, 2, 3, 2, 2],
        "R2": [1, 2, 1, 3, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3],
        "R3": 4,
    }
    non_renewable_resources = ["R3"]
    # Define mode details
    # Format: { task_id: { mode_id: { "duration": int, "resource_name": amount } } }
    mode_details = {
        "source": {1: {"duration": 0, "R1": 0, "R2": 0}},  # Source (Dummy)
        "task-1": {1: {"duration": 4, "R1": 1, "R2": 1}},
        "task-2": {1: {"duration": 1, "R1": 0, "R2": 1}},
        "task-3": {1: {"duration": 2, "R1": 0, "R2": 2}},
        "task-4": {
            1: {"duration": 2, "R1": 2, "R2": 1, "R3": 3},
            2: {"duration": 2, "R1": 2, "R2": 1, "R3": 5},
        },
        "sink": {1: {"duration": 0, "R1": 0, "R2": 0}},  # Sink (Dummy)
    }
    # Define precedence graph
    successors = {
        "source": ["task-1", "task-2"],  # First tasks: 1 and 2
        "task-1": ["task-3"],  # Task 1 must finish before 3
        "task-2": ["task-4"],  # Task 2 must finish before 4
        "task-3": ["sink"],  # Task 3 leads to Sink
        "task-4": ["sink"],  # Task 4 leads to Sink
        "sink": [],  # Sink has no successors
    }
    # Initialize the Problem
    problem = RcpspProblem(
        resources=resources,
        non_renewable_resources=non_renewable_resources,
        mode_details=mode_details,
        successors=successors,
        horizon=15,  # Maximum time allowed for the project
        source_task="source",
        sink_task="sink",
    )
    return problem


@pytest.fixture
def small_problem_no_calendar():
    resources = {"R1": 5, "R2": 3, "R3": 4}
    non_renewable_resources = ["R3"]
    # Define mode details
    # Format: { task_id: { mode_id: { "duration": int, "resource_name": amount } } }
    mode_details = {
        "source": {1: {"duration": 0, "R1": 0, "R2": 0}},  # Source (Dummy)
        "task-1": {1: {"duration": 4, "R1": 1, "R2": 1}},
        "task-2": {1: {"duration": 1, "R1": 0, "R2": 1}},
        "task-3": {1: {"duration": 2, "R1": 0, "R2": 2}},
        "task-4": {
            1: {"duration": 2, "R1": 2, "R2": 1, "R3": 3},
            2: {"duration": 2, "R1": 2, "R2": 1, "R3": 5},
        },
        "sink": {1: {"duration": 0, "R1": 0, "R2": 0}},  # Sink (Dummy)
    }
    # Define precedence graph
    successors = {
        "source": ["task-1", "task-2"],  # First tasks: 1 and 2
        "task-1": ["task-3"],  # Task 1 must finish before 3
        "task-2": ["task-4"],  # Task 2 must finish before 4
        "task-3": ["sink"],  # Task 3 leads to Sink
        "task-4": ["sink"],  # Task 4 leads to Sink
        "sink": [],  # Sink has no successors
    }
    # Initialize the Problem
    problem = RcpspProblem(
        resources=resources,
        non_renewable_resources=non_renewable_resources,
        mode_details=mode_details,
        successors=successors,
        horizon=15,  # Maximum time allowed for the project
        source_task="source",
        sink_task="sink",
    )
    return problem


def test_update_problem(small_problem):
    problem = small_problem

    assert problem.get_resource_max_capacity("R1") == 5
    assert len(problem.get_resource_availabilities("R1")) == 8

    problem.resources["R1"][1] = 7
    assert problem.get_resource_max_capacity("R1") == 5
    assert len(problem.get_resource_availabilities("R1")) == 8

    problem.update_problem()
    assert problem.get_resource_max_capacity("R1") == 7
    assert len(problem.get_resource_availabilities("R1")) == 9


def test_satisfy_renewable_nok(small_problem, caplog):
    problem = small_problem
    solution = RcpspSolution(
        problem=problem,
        rcpsp_permutation=[1, 0, 3, 2],
        fast=False,  # avoid overhead for first call to numba functions
    )  # => task-1 and task-4 together at time 4
    assert problem.satisfy(solution)

    # tweak calendar to invalidate the solution
    t = 4
    resource = "R1"
    problem.resources[resource][t] = 2
    problem.update_problem()
    problem.satisfy(solution)
    with caplog.at_level(logging.DEBUG):
        assert not problem.satisfy(solution)
    assert resource in caplog.text
    assert f"time {t}" in caplog.text


def test_satisfy_non_renewable_nok(small_problem):
    problem = small_problem
    solution = RcpspSolution(
        problem=problem,
        rcpsp_permutation=[1, 0, 3, 2],
        fast=False,  # avoid overhead for first call to numba functions
    )  # => task-1 and task-4 together at time 4
    assert problem.satisfy(solution)

    # change mode to violate non-renewable constraint
    solution = RcpspSolution(
        problem=problem,
        rcpsp_permutation=[1, 0, 3, 2],
        rcpsp_modes=[1, 1, 1, 2],
        fast=True,  # avoid overhead for first call to numba functions
    )
    assert not problem.satisfy(solution)


def test_satisfy_non_renewable_no_calendar_nok(small_problem_no_calendar):
    problem = small_problem_no_calendar
    solution = RcpspSolution(
        problem=problem,
        rcpsp_permutation=[1, 0, 3, 2],
        fast=False,  # avoid overhead for first call to numba functions
    )  # => task-1 and task-4 together at time 4
    assert problem.satisfy(solution)

    # change mode to violate non-renewable constraint
    solution = RcpspSolution(
        problem=problem,
        rcpsp_permutation=[1, 0, 3, 2],
        rcpsp_modes=[1, 1, 1, 2],
        fast=True,  # avoid overhead for first call to numba functions
    )
    assert not problem.satisfy(solution)
