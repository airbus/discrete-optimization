from typing import Dict, List, Set

from discrete_optimization.rcpsp.plots.rcpsp_utils_preemptive import plot_ressource_view
from discrete_optimization.rcpsp.solver.ls_solver import LS_SOLVER, LS_RCPSP_Solver
from discrete_optimization.rcpsp_multiskill.plots.plot_solution import (
    plot_resource_individual_gantt_preemptive,
    plt,
)
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import (
    Employee,
    MS_RCPSPModel,
    MS_RCPSPModel_Variant,
    MS_RCPSPSolution,
    MS_RCPSPSolution_Preemptive_Variant,
    MS_RCPSPSolution_Variant,
    SkillDetail,
    SpecialConstraintsDescription,
    TaskDetails,
    TaskDetailsPreemptive,
    compute_constraints_details,
)
from discrete_optimization.rcpsp_multiskill.solvers.cp_solvers import (
    CP_MS_MRCPSP_MZN_PARTIAL_PREEMPTIVE,
    ParametersCP,
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
        preemptive=True,
        preemptive_indicator={t: True for t in mode_details},
        horizon_multiplier=1,
        never_releasable_resources={"R1", "R2"},
        always_releasable_resources={"R3"},
    )
    return model


def create_toy_problem_paper():
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
    # partial_preemptive_data["A3"][1]["R1"] = False
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


def run_partial_preemptive():

    model = create_toy_problem_paper()
    model_variant: MS_RCPSPModel_Variant = model.to_variant_model()
    import time

    for j in range(100):
        t = time.time()
        dummy_solution = model_variant.get_dummy_solution(preemptive=True)
        print(model.evaluate(dummy_solution))
        print(compute_constraints_details(dummy_solution, model.special_constraints))
        t_end = time.time()
        print(t_end - t, " sec ")
    plot_resource_individual_gantt_preemptive(
        rcpsp_model=model, rcpsp_sol=dummy_solution
    )
    plot_ressource_view(rcpsp_model=model, rcpsp_sol=dummy_solution)
    plt.show()
    cp_solver = CP_MS_MRCPSP_MZN_PARTIAL_PREEMPTIVE(rcpsp_model=model)
    cp_solver.init_model(
        max_time=100,
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
    plot_resource_individual_gantt_preemptive(
        rcpsp_model=model, rcpsp_sol=result_storage.get_last_best_solution()[0]
    )
    plot_ressource_view(
        rcpsp_model=model, rcpsp_sol=result_storage.get_last_best_solution()[0]
    )
    plt.show()


def try_ls():
    model = create_toy_problem_paper()
    model_variant: MS_RCPSPModel_Variant = model.to_variant_model()
    solver = LS_RCPSP_Solver(model=model_variant, ls_solver=LS_SOLVER.SA)
    result_storage = solver.solve(nb_iteration_max=2000, verbose=True)
    plot_resource_individual_gantt_preemptive(
        rcpsp_model=model, rcpsp_sol=result_storage.get_last_best_solution()[0]
    )
    plot_ressource_view(
        rcpsp_model=model, rcpsp_sol=result_storage.get_last_best_solution()[0]
    )
    plt.show()


if __name__ == "__main__":
    try_ls()
