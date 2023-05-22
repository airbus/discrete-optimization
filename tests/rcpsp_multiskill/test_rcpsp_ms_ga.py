#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random
from typing import Dict, List, Set

import numpy as np
import pytest

from discrete_optimization.generic_tools.do_problem import ObjectiveHandling
from discrete_optimization.generic_tools.ea.alternating_ga import AlternatingGa
from discrete_optimization.generic_tools.ea.ga import DeapCrossover, DeapMutation, Ga
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import (
    Employee,
    MS_RCPSPModel,
    MS_RCPSPModel_Variant,
    SkillDetail,
)
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill_parser import (
    get_data_available,
    parse_file,
)


@pytest.fixture
def random_seed():
    random.seed(0)
    np.random.seed(0)


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
        4: Employee(
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


def test_alternating_ga_specific_mode_arity(random_seed):
    msrcpsp_model = create_toy_msrcpsp_variant()
    files = [f for f in get_data_available() if "100_5_64_9.def" in f]
    msrcpsp_model, new_tame_to_original_task_id = parse_file(files[0], max_horizon=2000)
    msrcpsp_model = msrcpsp_model.to_variant_model()

    total_evals = 50
    evals_per_ga_runs_perm = 0.33 * total_evals
    evals_per_ga_runs_modes = 0.33 * total_evals
    evals_per_ga_runs_resource_perm = 0.34 * total_evals

    mode_mutation = DeapMutation.MUT_UNIFORM_INT
    task_permutation_mutation = DeapMutation.MUT_SHUFFLE_INDEXES
    resource_permutation_mutation = DeapMutation.MUT_SHUFFLE_INDEXES

    # Initialise the task permutation that will be used to first search through the modes
    initial_task_permutation = [i for i in range(msrcpsp_model.n_jobs_non_dummy)]
    msrcpsp_model.set_fixed_task_permutation(initial_task_permutation)

    # Initialise the resource permutation that will be used to first search through the modes
    initial_resource_permutation = [
        i for i in range(len(msrcpsp_model.tasks) * len(msrcpsp_model.employees.keys()))
    ]
    msrcpsp_model.set_fixed_priority_worker_per_task_from_permutation(
        initial_resource_permutation
    )

    # Run a GA for evals_per_ga_runs evals on modes
    ga_solver = Ga(
        msrcpsp_model,
        encoding="modes_arity_fix_from_0",
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=["makespan"],
        objective_weights=[-1],
        mutation=mode_mutation,
        max_evals=evals_per_ga_runs_modes,
    )
    tmp_sol = ga_solver.solve().get_best_solution()
    # Fix the resulting modes
    msrcpsp_model.set_fixed_modes(tmp_sol.modes_vector)

    # Run a GA for evals_per_ga_runs evals on permutation
    ga_solver = Ga(
        msrcpsp_model,
        encoding="priority_list_task",
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=["makespan"],
        objective_weights=[-1],
        mutation=task_permutation_mutation,
        max_evals=evals_per_ga_runs_perm,
    )
    tmp_sol = ga_solver.solve().get_best_solution()

    # Fix the resulting permutation
    msrcpsp_model.set_fixed_task_permutation(tmp_sol.priority_list_task)

    # Run a GA for evals_per_ga_runs evals on permutation resource
    ga_solver = Ga(
        msrcpsp_model,
        encoding="priority_worker_per_task_perm",
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=["makespan"],
        objective_weights=[-1],
        mutation=resource_permutation_mutation,
        max_evals=evals_per_ga_runs_resource_perm,
    )
    sol = ga_solver.solve().get_best_solution()

    # Fix the resulting permutation
    msrcpsp_model.set_fixed_priority_worker_per_task(sol.priority_worker_per_task)

    assert msrcpsp_model.satisfy(sol)
    msrcpsp_model.evaluate(tmp_sol)


def test_alternating_ga_specific_mode_arity_single_solver(random_seed):
    msrcpsp_model = create_toy_msrcpsp_variant()

    total_evals = 1000

    sub_evals = [50, 50, 50]

    ga_solver = AlternatingGa(
        msrcpsp_model,
        encodings=[
            "modes_arity_fix_from_0",
            "priority_list_task",
            "priority_worker_per_task_perm",
        ],
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=["makespan"],
        objective_weights=[-1],
        mutations=[
            DeapMutation.MUT_UNIFORM_INT,
            DeapMutation.MUT_SHUFFLE_INDEXES,
            DeapMutation.MUT_SHUFFLE_INDEXES,
        ],
        crossovers=[
            DeapCrossover.CX_ONE_POINT,
            DeapCrossover.CX_PARTIALY_MATCHED,
            DeapCrossover.CX_PARTIALY_MATCHED,
        ],
        max_evals=total_evals,
        sub_evals=sub_evals,
    )

    tmp_sol = ga_solver.solve().get_best_solution()
    assert msrcpsp_model.satisfy(tmp_sol)
    assert msrcpsp_model.evaluate(tmp_sol) == {"makespan": 30}
