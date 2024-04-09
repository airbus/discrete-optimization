#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Dict, List

import pytest

from discrete_optimization.generic_rcpsp_tools.ls_solver import (
    LS_SOLVER,
    LS_RCPSP_Solver,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.rcpsp.plots.rcpsp_utils_preemptive import plot_ressource_view
from discrete_optimization.rcpsp.special_constraints import (
    SpecialConstraintsDescription,
)
from discrete_optimization.rcpsp_multiskill.plots.plot_solution import (
    plot_resource_individual_gantt_preemptive,
)
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import (
    Employee,
    MS_RCPSPModel,
    MS_RCPSPModel_Variant,
    SkillDetail,
    compute_constraints_details,
)
from discrete_optimization.rcpsp_multiskill.solvers.cp_solvers import (
    CP_MS_MRCPSP_MZN_PARTIAL_PREEMPTIVE,
)


@pytest.fixture
def model():
    skills_set = {"l1", "l2", "l3", "l4"}
    resource_set = {"R1"}
    resources_availability = {"R1": [2] * 100}
    employee: Dict[int, Employee] = {
        1: Employee(
            dict_skill={
                "l1": SkillDetail(1.0, 1.0, 1.0),
                "l3": SkillDetail(1.0, 1.0, 1.0),
            },
            calendar_employee=[True] * 100,
        ),
        2: Employee(
            dict_skill={
                "l1": SkillDetail(1.0, 1.0, 1.0),
                "l2": SkillDetail(1.0, 1.0, 1.0),
                "l4": SkillDetail(1.0, 1.0, 1.0),
            },
            calendar_employee=[True] * 100,
        ),
    }

    employees_availability: List[int] = [2] * 1000
    mode_details = {
        "A0": {1: {"R1": 0, "duration": 0}},
        "A1": {1: {"R1": 1, "l1": 1, "duration": 5}},
        "A2": {1: {"R1": 1, "l3": 1, "l4": 1, "duration": 1}},
        "A3": {1: {"R1": 1, "l2": 1, "duration": 3}},
        "A4": {1: {"R1": 0, "l3": 1, "duration": 2}},
        "A5": {1: {"R1": 0, "duration": 0}},
    }
    successors: Dict[str, List[str]] = {
        "A0": ["A" + str(i) for i in range(1, 6)],
        "A1": ["A5"],
        "A2": ["A5"],
        "A3": ["A5"],
        "A4": ["A5"],
        "A5": [],
    }
    partial_preemptive_data = {
        t: {m: {"R1": True} for m in mode_details[t]} for t in mode_details
    }
    special_constraint_description = SpecialConstraintsDescription(
        start_times_window={"A2": (2, None), "A4": (5, None)},
        end_times_window={"A2": (None, 3)},
    )
    return MS_RCPSPModel(
        skills_set=skills_set,
        resources_set=resource_set,
        non_renewable_resources=set(),
        resources_availability=resources_availability,
        employees=employee,
        employees_availability=employees_availability,
        mode_details=mode_details,
        successors=successors,
        horizon=12,
        tasks_list=["A" + str(i) for i in range(6)],
        source_task="A0",
        sink_task="A5",
        preemptive=True,
        preemptive_indicator={"A1": True, "A2": False, "A3": True, "A4": False},
        special_constraints=special_constraint_description,
        partial_preemption_data=partial_preemptive_data,
    )


def test_partial_preemptive(model):
    model_variant: MS_RCPSPModel_Variant = model.to_variant_model()

    dummy_solution = model_variant.get_dummy_solution(preemptive=True)
    model.evaluate(dummy_solution)
    compute_constraints_details(dummy_solution, model.special_constraints)
    assert model.satisfy(dummy_solution)

    plot_resource_individual_gantt_preemptive(
        rcpsp_model=model, rcpsp_sol=dummy_solution
    )
    plot_ressource_view(rcpsp_model=model, rcpsp_sol=dummy_solution)
    cp_solver = CP_MS_MRCPSP_MZN_PARTIAL_PREEMPTIVE(problem=model)
    cp_solver.init_model(
        max_time=20,
        max_preempted=3,
        nb_preemptive=10,
        possibly_preemptive=[
            model.preemptive_indicator.get(t, True) for t in model.tasks_list
        ],
        partial_solution=model.special_constraints,
        add_partial_solution_hard_constraint=True,
        unit_usage_preemptive=True,
    )
    result_storage = cp_solver.solve(parameters_cp=ParametersCP.default())
    rcpsp_sol = result_storage.get_last_best_solution()[0]
    assert model.satisfy(rcpsp_sol)
    plot_resource_individual_gantt_preemptive(rcpsp_model=model, rcpsp_sol=rcpsp_sol)
    plot_ressource_view(rcpsp_model=model, rcpsp_sol=rcpsp_sol)


def test_ls(model):
    model_variant: MS_RCPSPModel_Variant = model.to_variant_model()
    solver = LS_RCPSP_Solver(problem=model_variant, ls_solver=LS_SOLVER.SA)
    result_storage = solver.solve(nb_iteration_max=5000)
    rcpsp_sol = result_storage.get_last_best_solution()[0]
    assert model.satisfy(rcpsp_sol)
    plot_resource_individual_gantt_preemptive(rcpsp_model=model, rcpsp_sol=rcpsp_sol)
    plot_ressource_view(rcpsp_model=model, rcpsp_sol=rcpsp_sol)
