#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Dict, List, Set

from discrete_optimization.generic_tools.cp_tools import CPSolverName, ParametersCP
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import (
    Employee,
    MS_RCPSPModel_Variant,
    MS_RCPSPSolution,
    SkillDetail,
)
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.rcpsp_multiskill.solvers.cp_solvers import (
    CP_MS_MRCPSP_MZN,
    SearchStrategyMS_MRCPSP,
)


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
    employee = {
        1: Employee(
            dict_skill={
                "S1": SkillDetail(10, 0, 0),
                "S2": SkillDetail(10, 0, 0),
                "S3": SkillDetail(10, 0, 0),
            },
            calendar_employee=[True] * 100,
        ),
        2: Employee(
            dict_skill={"S2": SkillDetail(10, 0, 0), "S3": SkillDetail(10, 0, 0)},
            calendar_employee=[True] * 100,
        ),
        3: Employee(
            dict_skill={"S3": SkillDetail(10, 0, 0)}, calendar_employee=[True] * 100
        ),
    }
    index = 5
    for emp in sorted(employee):
        indexes = [index + 8 * i for i in range(10)] + [
            index + 1 + 8 * i for i in range(10)
        ]
        for i in indexes:
            employee[emp].calendar_employee[i] = True
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
        5: {1: {"R1": 2, "R2": 0, "R3": 0, "S1": 1, "duration": 5}},
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


def create_toy_v2():
    skills_set: Set[str] = {"S1", "S2", "S3"}
    resources_set: Set[str] = {"R1", "R2", "R3"}
    non_renewable_resources = set()
    resources_availability = {"R1": [200] * 100, "R2": [400] * 100, "R3": [300] * 100}

    employee: Dict[int, Employee] = {
        1: Employee(
            dict_skill={
                "S1": SkillDetail(10, 0, 0),
                "S2": SkillDetail(10, 0, 0),
                "S3": SkillDetail(10, 0, 0),
            },
            calendar_employee=[True] * 100,
        ),
        2: Employee(
            dict_skill={"S2": SkillDetail(10, 0, 0), "S3": SkillDetail(10, 0, 0)},
            calendar_employee=[True] * 100,
        ),
        3: Employee(
            dict_skill={"S3": SkillDetail(10, 0, 0)}, calendar_employee=[True] * 100
        ),
    }
    index = 5
    for emp in sorted(employee):
        indexes = [index + 8 * i for i in range(10)] + [
            index + 1 + 8 * i for i in range(10)
        ]
        for i in indexes:
            employee[emp].calendar_employee[i] = True
        index += 1

    employees_availability: List[int] = [3] * 1000
    mode_details: Dict[int, Dict[int, Dict[str, int]]] = {
        1: {1: {"R1": 0, "R2": 0, "R3": 0, "duration": 0}},
        2: {1: {"S1": 1, "S3": 1, "R1": 1, "R2": 0, "R3": 0, "duration": 2}},
        3: {1: {"S2": 1, "R1": 1, "R2": 2, "R3": 0, "duration": 4}},
        4: {1: {"S3": 1, "S2": 1, "R1": 2, "R2": 0, "R3": 0, "duration": 5}},
        5: {1: {"R1": 2, "R2": 0, "R3": 0, "S1": 1, "duration": 5}},
        6: {1: {"S3": 1, "S2": 1, "R1": 2, "R2": 0, "R3": 0, "duration": 6}},
        7: {
            1: {"S3": 1, "R1": 2, "R2": 0, "R3": 0, "duration": 1},
            2: {"R1": 2, "R2": 0, "R3": 0, "duration": 1},
        },
        8: {1: {"R1": 0, "R2": 0, "R3": 0, "duration": 0}},
    }

    successors: Dict[int, List[int]] = {
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
        7: [],
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


def test_cp_toy_model():
    model_msrcpsp = create_toy_v2()

    cp_model = CP_MS_MRCPSP_MZN(
        problem=model_msrcpsp, cp_solver_name=CPSolverName.GECODE
    )
    cp_model.init_model(
        add_calendar_constraint_unit=False,
        fake_tasks=True,
        output_type=True,
        exact_skills_need=False,
    )
    parameters_cp = ParametersCP.default()
    parameters_cp.intermediate_solution = True
    parameters_cp.time_limit = 200
    result_storage = cp_model.solve(parameters_cp=parameters_cp)
    solution: MS_RCPSPSolution = result_storage.get_best_solution()
    assert model_msrcpsp.satisfy(solution)


def test_cp_imopse():

    file = [f for f in get_data_available() if "100_5_20_9_D3.def" in f][0]
    model_msrcpsp, new_tame_to_original_task_id = parse_file(file, max_horizon=1000)
    cp_model = CP_MS_MRCPSP_MZN(
        problem=model_msrcpsp,
        one_ressource_per_task=True,
        cp_solver_name=CPSolverName.CHUFFED,
    )
    cp_model.init_model(
        model_type="multi-calendar",
        add_calendar_constraint_unit=False,
        fake_tasks=False,
        add_objective_makespan=True,
        ignore_sec_objective=True,
        output_type=True,
        max_time=500,  # here you put makespan constraint. by default,
        # would use model_msrcpsp.horizon if not provided.
        search_strategy=SearchStrategyMS_MRCPSP.PRIORITY_SEARCH_START_UNIT_USED,
        exact_skills_need=False,
    )
    parameters_cp = ParametersCP.default()
    parameters_cp.free_search = False
    # With parameters_cp.free_search=True you get fast results but quite bad !
    # but can be a good thing to be used in findmus algo
    # with free_search=False, you get first results after 10 seconds or so, but good quality.
    parameters_cp.intermediate_solution = True
    parameters_cp.time_limit = 30
    result_storage = cp_model.solve(parameters_cp=parameters_cp)
    solution: MS_RCPSPSolution = result_storage.get_best_solution()
    assert model_msrcpsp.satisfy(solution)
    model_msrcpsp.evaluate(solution)


if __name__ == "__main__":
    test_cp_imopse()
