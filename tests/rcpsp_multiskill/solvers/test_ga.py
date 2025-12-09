#  Copyright (c) 2022-2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random

import numpy as np
import pytest

from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveHandling,
    ParamsObjectiveFunction,
)
from discrete_optimization.generic_tools.ea.alternating_ga import AlternatingGa
from discrete_optimization.generic_tools.ea.ga import DeapCrossover, DeapMutation
from discrete_optimization.generic_tools.ea.ga_tools import ParametersAltGa
from discrete_optimization.rcpsp_multiskill.problem import (
    Employee,
    MultiskillRcpspProblem,
    SkillDetail,
    VariantMultiskillRcpspProblem,
)
from discrete_optimization.rcpsp_multiskill.solvers.ga import GaMultiskillRcpspSolver


@pytest.fixture
def random_seed():
    random.seed(0)
    np.random.seed(0)


def create_toy_msrcpsp():
    skills_set: set[str] = {"S1", "S2", "S3"}
    resources_set: set[str] = {"R1", "R2", "R3"}
    non_renewable_resources = set()
    resources_availability = {"R1": [2] * 100, "R2": [4] * 100, "R3": [3] * 100}
    employee: dict[int, Employee] = {
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

    employees_availability: list[int] = [3] * 1000
    mode_details: dict[int, dict[int, dict[str, int]]] = {
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
    successors: dict[int, list[int]] = {
        1: [2, 3],
        2: [5],
        3: [4],
        4: [5],
        5: [6, 7],
        6: [8],
        7: [8],
        8: [],
    }

    model = MultiskillRcpspProblem(
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
    skills_set: set[str] = {"S1", "S2", "S3"}
    resources_set: set[str] = {"R1", "R2", "R3"}
    non_renewable_resources = set()
    resources_availability = {"R1": [2] * 100, "R2": [4] * 100, "R3": [3] * 100}
    employee: dict[int, Employee] = {
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

    employees_availability: list[int] = [3] * 1000
    mode_details: dict[int, dict[int, dict[str, int]]] = {
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
    successors: dict[int, list[int]] = {
        1: [2, 3],
        2: [5],
        3: [4],
        4: [5],
        5: [6, 7],
        6: [8],
        7: [8],
        8: [],
    }

    model = VariantMultiskillRcpspProblem(
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


def test_ga_multiskill_rcpsp_solver(random_seed):
    msrcpsp_problem = create_toy_msrcpsp_variant()
    params_ga = ParametersAltGa.default_msrcpsp()
    params_ga.max_evals = 300
    params_ga.sub_evals = [50, 50, 50]
    solver = GaMultiskillRcpspSolver(problem=msrcpsp_problem)
    sol = solver.solve().get_best_solution()
    assert sol is not None


def test_alternating_ga(random_seed):
    msrcpsp_problem = create_toy_msrcpsp_variant()

    total_evals = 1000

    sub_evals = [50, 50, 50]

    params_objective_function = ParamsObjectiveFunction(
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=["makespan"],
        weights=[1],
        sense_function=ModeOptim.MINIMIZATION,
    )
    ga_solver = AlternatingGa(
        msrcpsp_problem,
        encodings=[
            "modes_vector_from0",
            "priority_list_task",
            "priority_worker_per_task_perm",
        ],
        params_objective_function=params_objective_function,
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
    assert msrcpsp_problem.satisfy(tmp_sol)
    assert msrcpsp_problem.evaluate(tmp_sol) == {"makespan": 30}
