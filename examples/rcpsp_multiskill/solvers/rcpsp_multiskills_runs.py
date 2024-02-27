#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random
import time
from typing import Dict, List, Set

import matplotlib.pyplot as plt

from discrete_optimization.rcpsp_multiskill.plots.plot_solution import (
    plot_resource_individual_gantt,
    plot_resource_individual_gantt_preemptive,
)
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import (
    Employee,
    MS_RCPSPModel,
    MS_RCPSPModel_Variant,
    SkillDetail,
    TaskDetails,
    TaskDetailsPreemptive,
)
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.rcpsp_multiskill.solvers.lp_model import (
    LP_Solver_MRSCPSP,
    MilpSolverName,
    ParametersMilp,
)


def run_lp_debug():
    skills_set: Set[str] = {"S1", "S2", "S3"}
    resources_set: Set[str] = {"R1", "R2", "R3"}
    non_renewable_resources = set()
    resources_availability = {"R1": [2] * 100, "R2": [4] * 100, "R3": [3] * 100}
    employee: Dict[int, Employee] = {
        1: Employee(
            dict_skill={"S1": SkillDetail(1.0, 1.0, 1.0)},
            calendar_employee=[True] * 1000,
        ),
        2: Employee(
            dict_skill={"S2": SkillDetail(1.0, 1.0, 1.0)},
            calendar_employee=[True] * 1000,
        ),
        3: Employee(
            dict_skill={"S3": SkillDetail(1.0, 1.0, 1.0)},
            calendar_employee=[True] * 1000,
        ),
    }
    employees_availability: List[int] = [3] * 1000
    mode_details: Dict[int, Dict[int, Dict[str, int]]] = {
        1: {1: {"R1": 0, "R2": 0, "R3": 0, "duration": 0}},
        2: {1: {"S1": 1, "R1": 2, "R2": 0, "R3": 0, "duration": 2}},
        3: {1: {"S2": 1, "R1": 1, "R2": 2, "R3": 0, "duration": 4}},
        4: {1: {"S3": 1, "R1": 2, "R2": 0, "R3": 0, "duration": 5}},
        5: {1: {"R1": 0, "R2": 0, "R3": 0, "duration": 0}},
    }
    successors: Dict[int, List[int]] = {1: [2, 3], 2: [5], 3: [4], 4: [5], 5: []}

    model = MS_RCPSPModel(
        skills_set=skills_set,
        resources_set=resources_set,
        non_renewable_resources=non_renewable_resources,
        resources_availability=resources_availability,
        employees=employee,
        employees_availability=employees_availability,
        mode_details=mode_details,
        successors=successors,
        horizon=100,
        horizon_multiplier=1,
    )
    lp_model = LP_Solver_MRSCPSP(problem=model, lp_solver=MilpSolverName.CBC)
    lp_model.init_model()
    result = lp_model.solve(parameters_milp=ParametersMilp.default())


def run_lp_debug_bis():
    skills_set: Set[str] = {"S1", "S2", "S3"}
    resources_set: Set[str] = {"R1", "R2", "R3"}
    non_renewable_resources = set()
    resources_availability = {"R1": [2] * 100, "R2": [4] * 100, "R3": [3] * 100}
    employee: Dict[int, Employee] = {
        1: Employee(
            dict_skill={"S1": SkillDetail(1.0, 1.0, 1.0)},
            calendar_employee=[True] * 100,
        ),
        2: Employee(
            dict_skill={"S2": SkillDetail(1.0, 1.0, 1.0)},
            calendar_employee=[True] * 100,
        ),
        3: Employee(
            dict_skill={"S3": SkillDetail(1.0, 1.0, 1.0)},
            calendar_employee=[True] * 100,
        ),
    }
    index = 5
    for emp in sorted(employee):
        indexes = [index + 8 * i for i in range(10)] + [
            index + 1 + 8 * i for i in range(10)
        ]
        for i in indexes:
            employee[emp].calendar_employee[i] = False
        index += 1

    employees_availability: List[int] = [3] * 1000
    mode_details: Dict[int, Dict[int, Dict[str, int]]] = {
        1: {1: {"R1": 0, "R2": 0, "R3": 0, "duration": 0}},
        2: {
            1: {"S1": 1, "R1": 2, "R2": 0, "R3": 0, "duration": 2},
            2: {"S2": 1, "R1": 0, "R2": 0, "R3": 0, "duration": 3},
        },
        3: {1: {"S2": 1, "R1": 1, "R2": 2, "R3": 0, "duration": 4}},
        4: {1: {"S3": 1, "R1": 2, "R2": 0, "R3": 0, "duration": 5}},
        5: {1: {"R1": 2, "R2": 0, "R3": 0, "duration": 5}},
        6: {1: {"S3": 1, "S2": 1, "R1": 2, "R2": 0, "R3": 0, "duration": 5}},
        7: {
            1: {"S3": 1, "R1": 2, "R2": 0, "R3": 0, "duration": 1},
            2: {"R1": 2, "R2": 0, "R3": 0, "duration": 2},
        },
        8: {1: {"R1": 0, "R2": 0, "R3": 0, "duration": 0}},
    }
    successors: Dict[int, List[int]] = {
        1: [2, 3],
        2: [5],
        3: [4],
        4: [5],
        5: [6, 7],
        6: [8],
        7: [8],
        8: [],
    }

    model = MS_RCPSPModel(
        skills_set=skills_set,
        resources_set=resources_set,
        non_renewable_resources=non_renewable_resources,
        resources_availability=resources_availability,
        employees=employee,
        employees_availability=employees_availability,
        mode_details=mode_details,
        successors=successors,
        horizon=100,
        horizon_multiplier=1,
    )
    lp_model = LP_Solver_MRSCPSP(problem=model, lp_solver=MilpSolverName.CBC)
    lp_model.init_model()
    result = lp_model.solve(parameters_milp=ParametersMilp.default())
    best_solution = result.get_best_solution()
    print(model.evaluate(best_solution))
    print(model.satisfy(best_solution))


def create_toy_msrcpsp():
    skills_set: Set[str] = {"S1", "S2", "S3"}
    resources_set: Set[str] = {"R1", "R2", "R3"}
    non_renewable_resources = set()
    resources_availability = {"R1": [2] * 100, "R2": [4] * 100, "R3": [3] * 100}
    employee: Dict[int, Employee] = {
        1: Employee(
            dict_skill={"S1": SkillDetail(1.0, 1.0, 1.0)},
            calendar_employee=[True] * 100,
        ),
        2: Employee(
            dict_skill={"S2": SkillDetail(1.0, 1.0, 1.0)},
            calendar_employee=[True] * 100,
        ),
        3: Employee(
            dict_skill={"S3": SkillDetail(1.0, 1.0, 1.0)},
            calendar_employee=[True] * 100,
        ),
    }
    index = 5
    for emp in sorted(employee):
        indexes = [index + 8 * i for i in range(10)] + [
            index + 1 + 8 * i for i in range(10)
        ]
        for i in indexes:
            employee[emp].calendar_employee[i] = False
        index += 1

    employees_availability: List[int] = [3] * 1000
    mode_details: Dict[int, Dict[int, Dict[str, int]]] = {
        1: {1: {"R1": 0, "R2": 0, "R3": 0, "duration": 0}},
        2: {
            1: {"S1": 1, "R1": 2, "R2": 0, "R3": 0, "duration": 2},
            2: {"S2": 1, "R1": 0, "R2": 0, "R3": 0, "duration": 3},
        },
        3: {1: {"S2": 1, "R1": 1, "R2": 2, "R3": 0, "duration": 4}},
        4: {1: {"S3": 1, "R1": 2, "R2": 0, "R3": 0, "duration": 5}},
        5: {1: {"R1": 2, "R2": 0, "R3": 0, "duration": 5}},
        6: {1: {"S3": 1, "S2": 1, "R1": 2, "R2": 0, "R3": 0, "duration": 5}},
        7: {
            1: {"S3": 1, "R1": 2, "R2": 0, "R3": 0, "duration": 1},
            2: {"R1": 2, "R2": 0, "R3": 0, "duration": 2},
        },
        8: {1: {"R1": 0, "R2": 0, "R3": 0, "duration": 0}},
    }
    successors: Dict[int, List[int]] = {
        1: [2, 3],
        2: [5],
        3: [4],
        4: [5],
        5: [6, 7],
        6: [8],
        7: [8],
        8: [],
    }

    model = MS_RCPSPModel(
        skills_set=skills_set,
        resources_set=resources_set,
        non_renewable_resources=non_renewable_resources,
        resources_availability=resources_availability,
        employees=employee,
        employees_availability=employees_availability,
        mode_details=mode_details,
        successors=successors,
        horizon=100,
        horizon_multiplier=1,
    )
    return model


def create_toy_msrcpsp_variant():
    skills_set: Set[str] = {"S1", "S2", "S3"}
    resources_set: Set[str] = {"R1", "R2", "R3"}
    non_renewable_resources = set()
    resources_availability = {"R1": [2] * 100, "R2": [4] * 100, "R3": [3] * 100}
    employee: Dict[int, Employee] = {
        1: Employee(
            dict_skill={"S1": SkillDetail(1.0, 1.0, 1.0)},
            calendar_employee=[True] * 100,
        ),
        2: Employee(
            dict_skill={"S2": SkillDetail(1.0, 1.0, 1.0)},
            calendar_employee=[True] * 100,
        ),
        3: Employee(
            dict_skill={"S3": SkillDetail(1.0, 1.0, 1.0)},
            calendar_employee=[True] * 100,
        ),
    }
    index = 5
    for emp in sorted(employee):
        indexes = [index + 8 * i for i in range(10)] + [
            index + 1 + 8 * i for i in range(10)
        ]
        for i in indexes:
            employee[emp].calendar_employee[i] = False
        index += 1

    employees_availability: List[int] = [3] * 1000
    mode_details: Dict[int, Dict[int, Dict[str, int]]] = {
        1: {1: {"R1": 0, "R2": 0, "R3": 0, "duration": 0}},
        2: {
            1: {"S1": 1, "R1": 2, "R2": 0, "R3": 0, "duration": 2},
            2: {"S2": 1, "R1": 0, "R2": 0, "R3": 0, "duration": 3},
        },
        3: {1: {"S2": 1, "R1": 1, "R2": 2, "R3": 0, "duration": 4}},
        4: {1: {"S3": 1, "R1": 2, "R2": 0, "R3": 0, "duration": 5}},
        5: {1: {"R1": 2, "R2": 0, "R3": 0, "duration": 5}},
        6: {1: {"S3": 1, "S2": 1, "R1": 2, "R2": 0, "R3": 0, "duration": 5}},
        7: {
            1: {"S3": 1, "R1": 2, "R2": 0, "R3": 0, "duration": 1},
            2: {"R1": 2, "R2": 0, "R3": 0, "duration": 2},
        },
        8: {1: {"R1": 0, "R2": 0, "R3": 0, "duration": 0}},
    }
    successors: Dict[int, List[int]] = {
        1: [2, 3],
        2: [5],
        3: [4],
        4: [5],
        5: [6, 7],
        6: [8],
        7: [8],
        8: [],
    }

    model = MS_RCPSPModel_Variant(
        skills_set=skills_set,
        resources_set=resources_set,
        non_renewable_resources=non_renewable_resources,
        resources_availability=resources_availability,
        employees=employee,
        employees_availability=employees_availability,
        mode_details=mode_details,
        successors=successors,
        horizon=100,
        horizon_multiplier=1,
    )
    return model


def sgs_debug(model=None):
    if model is None:
        model = create_toy_msrcpsp_variant()
    dummy_solution = model.get_dummy_solution()
    print("dummy_solution.priority_list_task: ", dummy_solution.priority_list_task)
    print(
        "dummy_solution.priority_worker_per_task: ",
        dummy_solution.priority_worker_per_task,
    )
    print("dummy_solution.modes_vector: ", dummy_solution.modes_vector)
    print(model.evaluate(dummy_solution))
    print(model.satisfy(dummy_solution))
    timesgs2 = int(dummy_solution.schedule[model.sink_task]["end_time"] / 2)
    finished = set(
        [
            t
            for t in dummy_solution.schedule
            if dummy_solution.schedule[t]["end_time"] <= timesgs2
        ]
    )
    completed = {
        t: TaskDetails(
            dummy_solution.schedule[t]["start_time"],
            dummy_solution.schedule[t]["end_time"],
            resource_units_used=list(dummy_solution.employee_usage.get(t, {}).keys()),
        )
        for t in finished
    }
    ongoing = {
        t: TaskDetails(
            dummy_solution.schedule[t]["start_time"],
            dummy_solution.schedule[t]["end_time"],
            resource_units_used=list(dummy_solution.employee_usage.get(t, {}).keys()),
        )
        for t in dummy_solution.schedule
        if dummy_solution.schedule[t]["start_time"]
        <= timesgs2
        < dummy_solution.schedule[t]["end_time"]
    }
    print("completed: ", completed)
    print("ongoing: ", ongoing)
    print("current_time: ", timesgs2)

    for i in range(1):
        t = time.time()
        dummy_solution.run_sgs_partial(
            current_t=timesgs2,
            completed_tasks=completed,
            scheduled_tasks_start_times=ongoing,
        )
    for o in ongoing:
        print(
            dummy_solution.schedule[o],
            dummy_solution.employee_usage.get(o, {}),
            ongoing[o],
        )
    for o in completed:
        print("completed")
        print(
            dummy_solution.schedule[o],
            dummy_solution.employee_usage.get(o, {}),
            completed[o],
        )
    print(model.satisfy(dummy_solution))
    print(model.evaluate(dummy_solution))

    plot_resource_individual_gantt(rcpsp_model=model, rcpsp_sol=dummy_solution)
    plt.show()


def sgs_debug_imopse():
    files = get_data_available()
    files = [f for f in get_data_available() if "100_5_22_15.def" in f]
    random.shuffle(files)
    model_msrcpsp, new_tame_to_original_task_id = parse_file(files[0], max_horizon=2000)
    model_msrcpsp = model_msrcpsp.to_variant_model()
    sgs_debug(model_msrcpsp)
    print(
        sum(
            [
                model_msrcpsp.mode_details[t][1]["duration"]
                for t in model_msrcpsp.mode_details
            ]
        )
    )


def sgs_debug_preemptive(model=None):
    if model is None:
        model = create_toy_msrcpsp_variant()
    dummy_solution = model.get_dummy_solution(preemptive=True)
    print("dummy_solution.priority_list_task: ", dummy_solution.priority_list_task)
    print(
        "dummy_solution.priority_worker_per_task: ",
        dummy_solution.priority_worker_per_task,
    )
    print("dummy_solution.modes_vector: ", dummy_solution.modes_vector)
    print(model.evaluate(dummy_solution))
    print(model.satisfy(dummy_solution))

    timesgs2 = int(dummy_solution.get_end_time(model.sink_task) / 2)
    finished = set(
        [
            t
            for t in dummy_solution.schedule
            if dummy_solution.schedule[t]["ends"][-1] <= timesgs2
        ]
    )
    completed = {}
    for t in finished:
        print(t)
        print(dummy_solution.get_start_times_list(t))
        print(
            [
                list(
                    dummy_solution.employee_usage.get(
                        t,
                        [
                            {}
                            for m in range(len(dummy_solution.get_start_times_list(t)))
                        ],
                    )[k].keys()
                )
                for k in range(len(dummy_solution.get_start_times_list(t)))
            ]
        )
        completed[t] = TaskDetailsPreemptive(
            dummy_solution.get_start_times_list(t),
            dummy_solution.get_end_times_list(t),
            resource_units_used=[
                list(
                    dummy_solution.employee_usage.get(
                        t,
                        [
                            {}
                            for m in range(len(dummy_solution.get_start_times_list(t)))
                        ],
                    )[k].keys()
                )
                for k in range(len(dummy_solution.get_start_times_list(t)))
            ],
        )
        print(t)
    ongoing = {
        t: TaskDetailsPreemptive(
            dummy_solution.get_start_times_list(t),
            dummy_solution.get_end_times_list(t),
            resource_units_used=[
                list(
                    dummy_solution.employee_usage.get(
                        t,
                        [
                            {}
                            for m in range(len(dummy_solution.get_start_times_list(t)))
                        ],
                    )[k].keys()
                )
                for k in range(len(dummy_solution.employee_usage[t]))
            ],
        )
        for t in dummy_solution.schedule
        if dummy_solution.schedule[t]["starts"][0]
        <= timesgs2
        < dummy_solution.schedule[t]["ends"][-1]
    }

    for i in range(1):
        t = time.time()
        dummy_solution.run_sgs_partial(
            current_t=timesgs2,
            completed_tasks=completed,
            scheduled_tasks_start_times=ongoing,
        )
    for o in ongoing:
        print(
            dummy_solution.schedule[o],
            dummy_solution.employee_usage.get(o, {}),
            ongoing[o],
        )
    for o in completed:
        print("completed")
        print(
            dummy_solution.schedule[o],
            dummy_solution.employee_usage.get(o, {}),
            completed[o],
        )
    print(model.satisfy(dummy_solution))
    print(model.evaluate(dummy_solution))

    plot_resource_individual_gantt_preemptive(
        rcpsp_model=model, rcpsp_sol=dummy_solution
    )
    plt.show()


def sgs_debug_fast(model=None):
    if model is None:
        model = create_toy_msrcpsp_variant()
        files = get_data_available()
        files = [f for f in get_data_available() if "100_5_22_15.def" in f]
        model_msrcpsp, new_tame_to_original_task_id = parse_file(
            files[0], max_horizon=2000, one_unit_per_task=False, preemptive=True
        )
        model = model_msrcpsp.to_variant_model()
    dummy_solution = model.get_dummy_solution()
    other_dummy_solution = model.get_dummy_solution()
    print("dummy_solution.priority_list_task: ", dummy_solution.priority_list_task)
    print(
        "dummy_solution.priority_worker_per_task: ",
        dummy_solution.priority_worker_per_task,
    )
    print("dummy_solution.modes_vector: ", dummy_solution.modes_vector)
    print(model.evaluate(dummy_solution))
    print(model.satisfy(dummy_solution))

    for i in range(10):
        t = time.time()
        dummy_solution.do_recompute(fast=False)
        t_end = time.time()
        print(model.evaluate(dummy_solution))
        print(other_dummy_solution.employee_usage)
        print(t_end - t, " seconds for the python sgs ")

    for i in range(10):
        t = time.time()
        other_dummy_solution.do_recompute(fast=True)
        t_end = time.time()
        print(model.evaluate(other_dummy_solution))
        print(model.satisfy(other_dummy_solution))
        print(other_dummy_solution.employee_usage)
        print(t_end - t, " seconds for the numba sgs ")


if __name__ == "__main__":
    sgs_debug_fast()
