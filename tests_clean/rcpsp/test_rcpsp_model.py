import random
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
    assert not rcpsp_model.satisfy(rcpsp_sol)
    evaluation = rcpsp_model.evaluate(rcpsp_sol)


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
    assert rcpsp_model.satisfy(rcpsp_sol)


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
