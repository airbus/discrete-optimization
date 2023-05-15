#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random

import pytest

from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel
from discrete_optimization.rcpsp.rcpsp_model_preemptive import (
    RCPSPModelPreemptive,
    RCPSPSolutionPreemptive,
    get_rcpsp_modelp_preemptive,
)
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file
from discrete_optimization.rcpsp.rcpsp_solution import RCPSPSolution

files_rcpsp = get_data_available()
single_modes_files = [f for f in files_rcpsp if "sm" in f]
multi_modes_files = [f for f in files_rcpsp if "mm" in f]


@pytest.mark.parametrize("rcpsp_model_file", single_modes_files)
def test_single_mode(rcpsp_model_file):
    rcpsp_model: RCPSPModel = parse_file(rcpsp_model_file)
    rcpsp_model: RCPSPModelPreemptive = get_rcpsp_modelp_preemptive(rcpsp_model)
    assert rcpsp_model.is_rcpsp_multimode() is False
    assert rcpsp_model.is_varying_resource() is False
    # Create solution (mode = 1 for each task, identity permutation)
    permutation = [i for i in range(rcpsp_model.n_jobs_non_dummy)]
    random.shuffle(permutation)
    mode_list = [1 for i in range(rcpsp_model.n_jobs_non_dummy)]
    rcpsp_sol = RCPSPSolutionPreemptive(
        problem=rcpsp_model, rcpsp_permutation=permutation, rcpsp_modes=mode_list
    )
    evaluation = rcpsp_model.evaluate(rcpsp_sol)
    assert rcpsp_model.satisfy(rcpsp_sol)


def test_unfeasible_modes_solution():
    files = get_data_available()
    files = [f for f in files if "j1010_5.mm" in f]  # Multi-mode RCPSP
    file_path = files[0]
    rcpsp_model: RCPSPModel = parse_file(file_path)
    rcpsp_model: RCPSPModelPreemptive = get_rcpsp_modelp_preemptive(rcpsp_model)
    permutation = [i for i in range(rcpsp_model.n_jobs_non_dummy)]
    unfeasible_modes = [2, 1, 1, 1, 3, 1, 2, 1, 1, 3]
    rcpsp_sol = RCPSPSolutionPreemptive(
        problem=rcpsp_model, rcpsp_permutation=permutation, rcpsp_modes=unfeasible_modes
    )
    assert not rcpsp_sol.rcpsp_schedule_feasible
    assert not rcpsp_model.satisfy(rcpsp_sol)
    rcpsp_model.evaluate(rcpsp_sol)


def test_feasible_modes_solution():
    files = get_data_available()
    files = [f for f in files if "j1010_1.mm" in f]  # Multi-mode RCPSP
    file_path = files[0]
    rcpsp_model = parse_file(file_path)
    rcpsp_model: RCPSPModel = parse_file(file_path)
    rcpsp_model: RCPSPModelPreemptive = get_rcpsp_modelp_preemptive(rcpsp_model)
    permutation = [i for i in range(rcpsp_model.n_jobs_non_dummy)]
    feasible_modes = [1, 1, 3, 3, 2, 2, 3, 3, 3, 2]
    rcpsp_sol = RCPSPSolutionPreemptive(
        problem=rcpsp_model, rcpsp_permutation=permutation, rcpsp_modes=feasible_modes
    )
    assert rcpsp_model.satisfy(rcpsp_sol)


def create_partial_sgs_input(solution: RCPSPSolution, time_to_cut: int):
    finished = set(
        [t for t in solution.rcpsp_schedule if solution.get_end_time(t) <= time_to_cut]
    )
    completed = finished
    partial_schedule = {
        t: {
            "starts": [
                solution.rcpsp_schedule[t]["starts"][k]
                for k in range(len(solution.rcpsp_schedule[t]["starts"]))
                if solution.rcpsp_schedule[t]["starts"][k] <= time_to_cut
            ],
            "ends": [
                solution.rcpsp_schedule[t]["ends"][k]
                for k in range(len(solution.rcpsp_schedule[t]["ends"]))
                if solution.rcpsp_schedule[t]["starts"][k] <= time_to_cut
            ],
        }
        for t in solution.rcpsp_schedule
        if solution.rcpsp_schedule[t]["starts"][0] <= time_to_cut
    }
    return completed, partial_schedule


@pytest.mark.parametrize("rcpsp_model_file", single_modes_files)
def test_partial_sgs(rcpsp_model_file):
    rcpsp_model: RCPSPModel = parse_file(rcpsp_model_file)
    rcpsp_model: RCPSPModelPreemptive = get_rcpsp_modelp_preemptive(rcpsp_model)
    assert rcpsp_model.is_rcpsp_multimode() is False
    assert rcpsp_model.is_varying_resource() is False
    rcpsp_sol = rcpsp_model.get_dummy_solution()
    time_to_cut = int(rcpsp_sol.get_end_time(rcpsp_model.sink_task) / 3)
    rcpsp_sol_copy = rcpsp_sol.copy()
    rcpsp_sol_copy.rcpsp_schedule[rcpsp_model.sink_task]
    completed, partial_schedule = create_partial_sgs_input(
        solution=rcpsp_sol, time_to_cut=time_to_cut
    )
    for i in range(10):
        rcpsp_sol_copy.generate_schedule_from_permutation_serial_sgs_2(
            current_t=time_to_cut,
            partial_schedule=partial_schedule,
            completed_tasks=completed,
            do_fast=True,
        )
        rcpsp_sol_copy.generate_schedule_from_permutation_serial_sgs_2(
            current_t=time_to_cut,
            partial_schedule=partial_schedule,
            completed_tasks=completed,
            do_fast=False,
        )

    for o in partial_schedule:
        starts = rcpsp_sol_copy.get_start_times_list(o)
        ends = rcpsp_sol_copy.get_end_times_list(o)
        for i in range(min(len(starts), len(partial_schedule[o]["starts"]))):
            assert starts[i] == partial_schedule[o]["starts"][i]
            assert ends[i] == partial_schedule[o]["ends"][i]
    rcpsp_sol_copy.generate_schedule_from_permutation_serial_sgs_2(
        current_t=time_to_cut,
        partial_schedule=partial_schedule,
        completed_tasks=completed,
        do_fast=True,
    )
    for o in partial_schedule:
        starts = rcpsp_sol_copy.get_start_times_list(o)
        ends = rcpsp_sol_copy.get_end_times_list(o)
        for i in range(min(len(starts), len(partial_schedule[o]["starts"]))):
            assert starts[i] == partial_schedule[o]["starts"][i]
            assert ends[i] == partial_schedule[o]["ends"][i]


if __name__ == "__main__":
    test_partial_sgs(single_modes_files[0])
