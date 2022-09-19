#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random
from typing import Dict, Hashable, Tuple

import pytest

from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import (
    MS_RCPSPModel,
    MS_RCPSPSolution_Preemptive_Variant,
    MS_RCPSPSolution_Variant,
    TaskDetails,
    TaskDetailsPreemptive,
)
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill_parser import (
    get_data_available,
    parse_file,
)

files_rcpsp = get_data_available()


@pytest.mark.parametrize("rcpsp_model_file", files_rcpsp)
def test_multiskill(rcpsp_model_file):
    rcpsp_model: MS_RCPSPModel = parse_file(rcpsp_model_file)[0]
    rcpsp_model = rcpsp_model.to_variant_model()
    assert rcpsp_model.is_rcpsp_multimode() is False
    assert rcpsp_model.is_varying_resource() is False
    # Create solution (mode = 1 for each task, identity permutation)
    permutation = [i for i in range(rcpsp_model.n_jobs_non_dummy)]
    random.shuffle(permutation)
    mode_list = [1 for i in range(rcpsp_model.n_jobs_non_dummy)]
    # This is actually the dummy solution !
    rcpsp_sol = MS_RCPSPSolution_Variant(
        problem=rcpsp_model,
        priority_list_task=permutation,
        priority_worker_per_task=[
            [w for w in rcpsp_model.employees]
            for i in range(rcpsp_model.n_jobs_non_dummy)
        ],
        modes_vector=mode_list,
        fast=True,
    )
    rcpsp_model.evaluate(rcpsp_sol)
    assert rcpsp_model.satisfy(rcpsp_sol)


def create_task_details_classic(
    solution: MS_RCPSPSolution_Variant, time_to_cut: int
) -> Tuple[Dict[Hashable, TaskDetails], Dict[Hashable, TaskDetails]]:
    finished = set(
        [t for t in solution.schedule if solution.get_end_time(t) <= time_to_cut]
    )
    completed = {
        t: TaskDetails(
            solution.get_start_time(t),
            solution.get_end_time(t),
            resource_units_used=list(solution.employee_usage.get(t, {}).keys()),
        )
        for t in finished
    }
    ongoing = {
        t: TaskDetails(
            solution.get_start_time(t),
            solution.get_end_time(t),
            resource_units_used=list(solution.employee_usage.get(t, {}).keys()),
        )
        for t in solution.schedule
        if solution.get_start_time(t) <= time_to_cut < solution.get_end_time(t)
    }
    return completed, ongoing


def create_task_details_preemptive(
    solution: MS_RCPSPSolution_Preemptive_Variant, time_to_cut: int
) -> Tuple[
    Dict[Hashable, TaskDetailsPreemptive], Dict[Hashable, TaskDetailsPreemptive]
]:
    finished = set(
        [t for t in solution.schedule if solution.get_end_time(t) <= time_to_cut]
    )
    completed = {}
    for t in finished:
        completed[t] = TaskDetailsPreemptive(
            solution.get_start_times_list(t),
            solution.get_end_times_list(t),
            resource_units_used=[
                list(
                    solution.employee_usage.get(
                        t, [{} for m in range(len(solution.get_start_times_list(t)))]
                    )[k].keys()
                )
                for k in range(len(solution.get_start_times_list(t)))
            ],
        )
    ongoing = {
        t: TaskDetailsPreemptive(
            solution.get_start_times_list(t),
            solution.get_end_times_list(t),
            resource_units_used=[
                list(
                    solution.employee_usage.get(
                        t, [{} for m in range(len(solution.get_start_times_list(t)))]
                    )[k].keys()
                )
                for k in range(len(solution.employee_usage[t]))
            ],
        )
        for t in solution.schedule
        if solution.schedule[t]["starts"][0]
        <= time_to_cut
        < solution.schedule[t]["ends"][-1]
    }
    return completed, ongoing


@pytest.mark.parametrize("rcpsp_model_file", files_rcpsp)
@pytest.mark.parametrize("preemptive_version", [True, False])
def test_partial_sgs(rcpsp_model_file, preemptive_version):
    rcpsp_model: MS_RCPSPModel = parse_file(
        rcpsp_model_file, preemptive=preemptive_version
    )[0]
    rcpsp_model = rcpsp_model.to_variant_model()
    class_solution = (
        MS_RCPSPSolution_Variant
        if not preemptive_version
        else MS_RCPSPSolution_Preemptive_Variant
    )
    dummy_solution: class_solution = rcpsp_model.get_dummy_solution()
    dummy_solution.priority_list_task
    dummy_solution.priority_worker_per_task
    dummy_solution.modes_vector
    rcpsp_model.evaluate(dummy_solution)
    assert rcpsp_model.satisfy(dummy_solution)
    timesgs2 = int(dummy_solution.get_end_time(rcpsp_model.sink_task) / 2)

    for i in range(2):
        dummy_solution.do_recompute(fast=False)
        rcpsp_model.evaluate(dummy_solution)
        dummy_solution.do_recompute(fast=True)
        rcpsp_model.evaluate(dummy_solution)
    if preemptive_version:
        completed, ongoing = create_task_details_preemptive(
            solution=dummy_solution, time_to_cut=timesgs2
        )
    else:
        completed, ongoing = create_task_details_classic(
            solution=dummy_solution, time_to_cut=timesgs2
        )
    for i in range(2):
        dummy_solution.run_sgs_partial(
            current_t=timesgs2,
            completed_tasks=completed,
            scheduled_tasks_start_times=ongoing,
            fast=False,
        )
        dummy_solution.run_sgs_partial(
            current_t=timesgs2,
            completed_tasks=completed,
            scheduled_tasks_start_times=ongoing,
            fast=True,
        )
    if preemptive_version:
        for dict_ in [ongoing, completed]:
            for o in dict_:
                (
                    dummy_solution.schedule[o],
                    dummy_solution.employee_usage.get(o, {}),
                    dict_[o],
                )
                starts = dummy_solution.get_start_times_list(o)
                ends = dummy_solution.get_end_times_list(o)
                employees = dummy_solution.employee_usage.get(o, [{}])
                for i in range(len(starts)):
                    assert starts[i] == dict_[o].starts[i]
                    assert ends[i] == dict_[o].ends[i]
                for i in range(len(employees)):
                    assert all(
                        e in dict_[o].resource_units_used[i] for e in employees[i]
                    )

    else:
        for dict_ in [ongoing, completed]:
            for o in dict_:
                (
                    dummy_solution.schedule[o],
                    dummy_solution.employee_usage.get(o, {}),
                    dict_[o],
                )
                assert dummy_solution.get_start_time(o) == dict_[o].start
                assert dummy_solution.get_end_time(o) == dict_[o].end
                assert all(
                    e in dict_[o].resource_units_used
                    for e in dummy_solution.employee_usage.get(o, {})
                )
