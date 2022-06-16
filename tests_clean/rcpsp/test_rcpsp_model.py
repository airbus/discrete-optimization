import random
from math import isclose
from typing import Dict, Hashable, Tuple

import pytest
from discrete_optimization.rcpsp.rcpsp_model import (
    MultiModeRCPSPModel,
    RCPSPModel,
    RCPSPSolution,
    SingleModeRCPSPModel,
    TaskDetails,
)
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file

files_rcpsp = get_data_available()
single_modes_files = [f for f in files_rcpsp if "sm" in f]
multi_modes_files = [f for f in files_rcpsp if "mm" in f]


@pytest.mark.parametrize("rcpsp_model_file", single_modes_files)
def test_single_mode(rcpsp_model_file):
    rcpsp_model: RCPSPModel = parse_file(rcpsp_model_file)
    assert rcpsp_model.is_rcpsp_multimode() is False
    assert rcpsp_model.is_varying_resource() is False
    # Create solution (mode = 1 for each task, identity permutation)
    permutation = [i for i in range(rcpsp_model.n_jobs_non_dummy)]
    random.shuffle(permutation)
    mode_list = [1 for i in range(rcpsp_model.n_jobs_non_dummy)]
    rcpsp_sol = RCPSPSolution(
        problem=rcpsp_model, rcpsp_permutation=permutation, rcpsp_modes=mode_list
    )
    evaluation = rcpsp_model.evaluate(rcpsp_sol)
    assert rcpsp_model.satisfy(rcpsp_sol)
    assert rcpsp_sol.rcpsp_schedule_feasible
    assert rcpsp_sol.rcpsp_modes == mode_list


def test_non_existing_modes_solution():
    files = get_data_available()
    files = [f for f in files if "j301_1.sm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_model = parse_file(file_path)

    permutation = [i for i in range(rcpsp_model.n_jobs_non_dummy)]
    mode_list = [2 for i in range(rcpsp_model.n_jobs_non_dummy)]
    rcpsp_sol = RCPSPSolution(
        problem=rcpsp_model, rcpsp_permutation=permutation, rcpsp_modes=mode_list
    )
    assert rcpsp_sol.rcpsp_schedule_feasible
    assert rcpsp_sol.rcpsp_schedule == {
        1: {"start_time": 0, "end_time": 0},
        2: {"start_time": 0, "end_time": 4},
        3: {"start_time": 0, "end_time": 6},
        4: {"start_time": 4, "end_time": 7},
        5: {"start_time": 7, "end_time": 15},
        6: {"start_time": 4, "end_time": 9},
        7: {"start_time": 6, "end_time": 15},
        8: {"start_time": 7, "end_time": 9},
        9: {"start_time": 7, "end_time": 14},
        10: {"start_time": 7, "end_time": 16},
        11: {"start_time": 4, "end_time": 6},
        12: {"start_time": 9, "end_time": 15},
        13: {"start_time": 15, "end_time": 18},
        14: {"start_time": 15, "end_time": 24},
        15: {"start_time": 15, "end_time": 25},
        16: {"start_time": 25, "end_time": 31},
        17: {"start_time": 31, "end_time": 36},
        18: {"start_time": 18, "end_time": 21},
        19: {"start_time": 18, "end_time": 25},
        20: {"start_time": 21, "end_time": 23},
        21: {"start_time": 31, "end_time": 38},
        22: {"start_time": 36, "end_time": 38},
        23: {"start_time": 38, "end_time": 41},
        24: {"start_time": 41, "end_time": 44},
        25: {"start_time": 25, "end_time": 32},
        26: {"start_time": 36, "end_time": 44},
        27: {"start_time": 25, "end_time": 28},
        28: {"start_time": 41, "end_time": 48},
        29: {"start_time": 28, "end_time": 30},
        30: {"start_time": 44, "end_time": 46},
        31: {"start_time": 48, "end_time": 48},
        32: {"start_time": 48, "end_time": 48},
    }
    assert rcpsp_sol.rcpsp_modes == mode_list
    with pytest.raises(KeyError):
        rcpsp_model.satisfy(rcpsp_sol)


def test_unfeasible_modes_solution():
    files = get_data_available()
    files = [f for f in files if "j1010_5.mm" in f]  # Multi-mode RCPSP
    file_path = files[0]
    rcpsp_model = parse_file(file_path)
    permutation = [i for i in range(rcpsp_model.n_jobs_non_dummy)]
    unfeasible_modes = [2, 1, 1, 1, 3, 1, 2, 1, 1, 3]
    rcpsp_sol = RCPSPSolution(
        problem=rcpsp_model, rcpsp_permutation=permutation, rcpsp_modes=unfeasible_modes
    )
    assert not rcpsp_sol.rcpsp_schedule_feasible
    assert rcpsp_sol.rcpsp_modes == unfeasible_modes
    assert not rcpsp_model.satisfy(rcpsp_sol)
    evaluation = rcpsp_model.evaluate(rcpsp_sol)
    assert evaluation == {"makespan": 99999999, "mean_resource_reserve": 0}


def test_feasible_modes_solution():
    files = get_data_available()
    files = [f for f in files if "j1010_1.mm" in f]  # Multi-mode RCPSP
    file_path = files[0]
    rcpsp_model = parse_file(file_path)

    permutation = [i for i in range(rcpsp_model.n_jobs_non_dummy)]
    feasible_modes = [1, 1, 3, 3, 2, 2, 3, 3, 3, 2]

    rcpsp_sol = RCPSPSolution(
        problem=rcpsp_model, rcpsp_permutation=permutation, rcpsp_modes=feasible_modes
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
    assert rcpsp_model.satisfy(rcpsp_sol)
    assert isclose(rcpsp_sol.compute_mean_resource_reserve(), 0.5028790398735252)


def create_task_details_classic(
    solution: RCPSPSolution, time_to_cut: int
) -> Tuple[Dict[Hashable, TaskDetails], Dict[Hashable, TaskDetails]]:
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


@pytest.mark.parametrize("rcpsp_model_file", single_modes_files)
def test_partial_sgs(rcpsp_model_file):
    rcpsp_model: RCPSPModel = parse_file(rcpsp_model_file)
    assert rcpsp_model.is_rcpsp_multimode() is False
    assert rcpsp_model.is_varying_resource() is False
    # Create solution (mode = 1 for each task, identity permutation)
    permutation = [i for i in range(rcpsp_model.n_jobs_non_dummy)]
    random.shuffle(permutation)
    mode_list = [1 for i in range(rcpsp_model.n_jobs_non_dummy)]
    rcpsp_sol = RCPSPSolution(
        problem=rcpsp_model,
        rcpsp_permutation=permutation,
        rcpsp_modes=mode_list,
        fast=True,
    )
    time_to_cut = int(rcpsp_sol.get_end_time(rcpsp_model.sink_task) / 2)
    completed, ongoing = create_task_details_classic(
        solution=rcpsp_sol, time_to_cut=time_to_cut
    )
    scheduled_start_time = {o: ongoing[o].start for o in ongoing}
    rcpsp_sol_copy = rcpsp_sol.copy()
    rcpsp_sol_copy.rcpsp_schedule[rcpsp_model.sink_task]

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

    rcpsp_sol_copy.rcpsp_schedule[rcpsp_model.sink_task]
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
    rcpsp_sol_copy.rcpsp_schedule[rcpsp_model.sink_task]
