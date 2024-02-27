#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from copy import deepcopy

import pytest

from discrete_optimization.generic_tools.cp_tools import CPSolverName, ParametersCP
from discrete_optimization.rcpsp.plots.rcpsp_utils_preemptive import (
    plot_resource_individual_gantt,
    plot_ressource_view,
    plot_task_gantt,
)
from discrete_optimization.rcpsp.rcpsp_model_preemptive import (
    RCPSPModelPreemptive,
    get_rcpsp_modelp_preemptive,
)
from discrete_optimization.rcpsp.rcpsp_parser import (
    RCPSPModel,
    get_data_available,
    parse_file,
)
from discrete_optimization.rcpsp.solver.cp_solvers import (
    CP_MRCPSP_MZN_PREEMPTIVE,
    CP_RCPSP_MZN_PREEMPTIVE,
)


def create_alamano_preemptive_model() -> RCPSPModelPreemptive:
    resource_r1 = []
    for i in range(40):
        resource_r1 += [3, 3, 3, 0, 0]
    return RCPSPModelPreemptive(
        resources={"R1": resource_r1, "R2": [2] * 200, "R3": [2] * 200},
        non_renewable_resources=[],
        mode_details={
            "A0": {1: {"duration": 0}},
            "A1": {1: {"duration": 5, "R1": 3, "R2": 1}},
            "A2": {1: {"duration": 2, "R1": 1}},
            "A3": {1: {"duration": 3, "R2": 1, "R3": 1}},
            "A4": {1: {"duration": 4, "R1": 2}},
            "A5": {1: {"duration": 5, "R1": 2, "R2": 1, "R3": 2}},
            "A6": {1: {"duration": 4, "R1": 2, "R3": 1}},
            "A7": {1: {"duration": 7, "R2": 1}},
            "A8": {1: {"duration": 2, "R1": 2, "R2": 1}},
            "A9": {1: {"duration": 0}},
        },
        successors={
            "A0": ["A" + str(i) for i in range(1, 10)],
            "A1": ["A4", "A9"],
            "A2": ["A9"],
            "A3": ["A5", "A9"],
            "A4": ["A6", "A9"],
            "A5": ["A7", "A8", "A9"],
            "A6": ["A8", "A9"],
            "A7": ["A9"],
            "A8": ["A9"],
            "A9": [],
        },
        horizon=200,
        horizon_multiplier=1,
        tasks_list=["A" + str(i) for i in range(10)],
        source_task="A0",
        sink_task="A9",
    )


def load_psplib_preemptive_model_1() -> RCPSPModelPreemptive:
    files = get_data_available()
    files = [f for f in files if "j1201_1.sm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_model = parse_file(file_path)
    return RCPSPModelPreemptive(
        resources=rcpsp_model.resources,
        non_renewable_resources=rcpsp_model.non_renewable_resources,
        mode_details=rcpsp_model.mode_details,
        successors=rcpsp_model.successors,
        horizon=rcpsp_model.horizon,
        horizon_multiplier=1,
        tasks_list=None,
        source_task=None,
        sink_task=None,
        preemptive_indicator={k: False for k in range(rcpsp_model.n_jobs)},
        name_task=None,
    )


def load_psplib_preemptive_model_2(filename) -> RCPSPModelPreemptive:
    files = get_data_available()
    files = [f for f in files if filename in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_model: RCPSPModel = parse_file(file_path)
    return get_rcpsp_modelp_preemptive(rcpsp_model)


@pytest.mark.parametrize(
    "rcpsp_problem",
    [create_alamano_preemptive_model(), load_psplib_preemptive_model_1()],
)
def test_run_preemptive(rcpsp_problem):
    solution = rcpsp_problem.get_dummy_solution()

    solution.generate_schedule_from_permutation_serial_sgs(do_fast=False)
    eval_notfast = rcpsp_problem.evaluate(solution)
    solution.generate_schedule_from_permutation_serial_sgs(do_fast=True)
    eval_fast = rcpsp_problem.evaluate(solution)
    assert eval_fast == eval_notfast
    assert rcpsp_problem.satisfy(solution)

    previous_schedule = deepcopy(solution.rcpsp_schedule)
    timesgs2 = int(solution.rcpsp_schedule[rcpsp_problem.sink_task]["ends"][-1] / 3)
    finished = set(
        [
            t
            for t in solution.rcpsp_schedule
            if solution.rcpsp_schedule[t]["ends"][-1] <= timesgs2
        ]
    )
    completed = finished
    partial_schedule = {
        t: {
            "starts": [
                solution.rcpsp_schedule[t]["starts"][k]
                for k in range(len(solution.rcpsp_schedule[t]["starts"]))
                if solution.rcpsp_schedule[t]["starts"][k] <= timesgs2
            ],
            "ends": [
                solution.rcpsp_schedule[t]["ends"][k]
                for k in range(len(solution.rcpsp_schedule[t]["ends"]))
                if solution.rcpsp_schedule[t]["starts"][k] <= timesgs2
            ],
        }
        for t in solution.rcpsp_schedule
        if solution.rcpsp_schedule[t]["starts"][0] <= timesgs2
    }

    solution.generate_schedule_from_permutation_serial_sgs_2(
        current_t=timesgs2,
        partial_schedule=partial_schedule,
        completed_tasks=completed,
        do_fast=False,
    )
    eval_notfast = rcpsp_problem.evaluate(solution)
    solution.generate_schedule_from_permutation_serial_sgs_2(
        current_t=timesgs2,
        partial_schedule=partial_schedule,
        completed_tasks=completed,
        do_fast=True,
    )
    eval_fast = rcpsp_problem.evaluate(solution)
    assert eval_fast == eval_notfast
    assert rcpsp_problem.satisfy(solution)
    for o in solution.rcpsp_schedule:
        assert solution.rcpsp_schedule[o] == previous_schedule[o]
    for o in partial_schedule:
        assert solution.rcpsp_schedule[o] == partial_schedule[o]
    for o in completed:
        assert solution.rcpsp_schedule[o] == partial_schedule[o]


def test_preemptive_cp_alamano():
    rcpsp_problem = create_alamano_preemptive_model()
    solver = CP_RCPSP_MZN_PREEMPTIVE(
        problem=rcpsp_problem, cp_solver_name=CPSolverName.CHUFFED
    )
    solver.init_model(
        output_type=True,
        model_type="single-preemptive-calendar",
        nb_preemptive=7,
        possibly_preemptive=[True for k in rcpsp_problem.tasks_list],
        max_preempted=20,
    )
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 100
    result_store = solver.solve(parameters_cp=parameters_cp)
    best_solution = result_store.get_best_solution()

    assert rcpsp_problem.satisfy(best_solution)
    rcpsp_problem.evaluate(best_solution)

    plot_ressource_view(rcpsp_problem, best_solution)
    plot_resource_individual_gantt(rcpsp_problem, best_solution)
    plot_task_gantt(rcpsp_problem, best_solution)


def test_preemptive_multimode_cp_alamano():
    rcpsp_problem = create_alamano_preemptive_model()
    solver = CP_MRCPSP_MZN_PREEMPTIVE(
        problem=rcpsp_problem, cp_solver_name=CPSolverName.CHUFFED
    )
    solver.init_model(
        output_type=True,
        model_type="multi-preemptive-calendar",
        nb_preemptive=7,
        possibly_preemptive=[True for k in rcpsp_problem.tasks_list],
        max_preempted=20,
    )
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 100
    result_store = solver.solve(parameters_cp=parameters_cp)
    best_solution = result_store.get_best_solution()

    assert rcpsp_problem.satisfy(best_solution)
    rcpsp_problem.evaluate(best_solution)
    plot_ressource_view(rcpsp_problem, best_solution)
    plot_resource_individual_gantt(rcpsp_problem, best_solution)
    plot_task_gantt(rcpsp_problem, best_solution)


def test_preemptive_cp_psplib():
    rcpsp_problem = load_psplib_preemptive_model_2("j601_5.sm")
    solver = CP_RCPSP_MZN_PREEMPTIVE(
        problem=rcpsp_problem, cp_solver_name=CPSolverName.CHUFFED
    )
    solver.init_model(
        output_type=True,
        model_type="single-preemptive",
        possibly_preemptive=[True for t in rcpsp_problem.tasks_list],
        nb_preemptive=4,
        max_preempted=20,
    )
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 100
    result_store = solver.solve(parameters_cp=parameters_cp)
    best_solution = result_store.get_best_solution()
    assert rcpsp_problem.satisfy(best_solution)
    rcpsp_problem.evaluate(best_solution)
    plot_ressource_view(rcpsp_problem, best_solution)
    plot_resource_individual_gantt(rcpsp_problem, best_solution)
    plot_task_gantt(rcpsp_problem, best_solution)


def test_preemptive_multimode_cp_psplib():
    rcpsp_problem = load_psplib_preemptive_model_2("j1010_5.mm")
    solver = CP_MRCPSP_MZN_PREEMPTIVE(
        problem=rcpsp_problem, cp_solver_name=CPSolverName.CHUFFED
    )
    solver.init_model(
        output_type=True,
        model_type="multi-preemptive",
        possibly_preemptive=[True for t in rcpsp_problem.tasks_list],
        nb_preemptive=4,
        max_preempted=20,
    )
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 100
    result_store = solver.solve(parameters_cp=parameters_cp)
    best_solution = result_store.get_best_solution()
    assert rcpsp_problem.satisfy(best_solution)
    rcpsp_problem.evaluate(best_solution)
    plot_ressource_view(rcpsp_problem, best_solution)
    plot_resource_individual_gantt(rcpsp_problem, best_solution)
    plot_task_gantt(rcpsp_problem, best_solution)


def test_preeemptive_sgs():
    rcpsp_problem = load_psplib_preemptive_model_2("j601_5.sm")
    rcpsp_problem.duration_subtask = {t: (True, 2) for t in rcpsp_problem.tasks_list}
    rcpsp_problem.any_duration_subtask_limited = True
    rcpsp_problem.update_function()
    solution = rcpsp_problem.get_dummy_solution()
    assert solution.get_max_preempted() == 3
    assert solution.get_nb_task_preemption() == 4
    assert solution.get_end_time(rcpsp_problem.sink_task) == 77


if __name__ == "__main__":
    test_preeemptive_sgs()
