import os
from typing import Dict, List, Set

# from tests.rcpsp_multiskills.solvers.instance_creator import create_ms_rcpsp_demo
from discrete_optimization.rcpsp.solver.ls_solver import LS_SOLVER, LS_RCPSP_Solver
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import (
    Employee,
    MS_RCPSPModel,
    MS_RCPSPModel_Variant,
    MS_RCPSPSolution,
    MS_RCPSPSolution_Variant,
    SkillDetail,
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


def run_ls():
    model = create_toy_msrcpsp()
    # model, model_rcpsp = create_ms_rcpsp_demo()
    model = model.to_variant_model()
    solver = LS_RCPSP_Solver(model=model, ls_solver=LS_SOLVER.SA)
    result = solver.solve(nb_iteration_max=100000)
    solution: MS_RCPSPSolution = result.get_best_solution()
    print("Evaluation ", model.evaluate(solution))
    print("Satisfaction ", model.satisfy(solution))


def run_ls_imopse():
    # model = create_toy_msrcpsp()
    from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill_parser import (
        get_data_available,
        parse_file,
    )

    file = [f for f in get_data_available() if "100_5_22_15.def" in f][0]
    model, name_task = parse_file(file, max_horizon=1000)
    model = model.to_variant_model()
    # model.preemptive = True
    solver = LS_RCPSP_Solver(model=model, ls_solver=LS_SOLVER.SA)
    result = solver.solve(nb_iteration_max=1)
    solution: MS_RCPSPSolution = result.get_best_solution()
    # write_solution(solution=solution,
    #                new_tame_to_original_task_id=name_task,
    #                file_path=os.path.join(folder_to_do_solution,
    #                                       os.path.basename(file) + "_ls_test.sol"))
    print("Evaluation ", model.evaluate(solution))
    print("Satisfaction ", model.satisfy(solution))
    from discrete_optimization.rcpsp_multiskill.plots.plot_solution import (
        plot_resource_individual_gantt,
        plt,
    )

    plot_resource_individual_gantt(rcpsp_model=model, rcpsp_sol=solution)
    plt.show()


if __name__ == "__main__":
    run_ls_imopse()