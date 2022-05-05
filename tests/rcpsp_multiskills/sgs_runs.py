from typing import Dict, List, Set

from discrete_optimization.rcpsp.solver.ls_solver import LS_SOLVER, LS_RCPSP_Solver
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import (
    Employee,
    MS_RCPSPModel,
    MS_RCPSPModel_Variant,
    MS_RCPSPSolution,
    MS_RCPSPSolution_Preemptive_Variant,
    MS_RCPSPSolution_Variant,
    SkillDetail,
    TaskDetails,
    TaskDetailsPreemptive,
)
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill_parser import (
    folder_to_do_only_cp_solution,
    folder_to_do_solution,
    get_data_available,
    get_results_do,
    get_results_do_cp,
    parse_file,
    write_solution,
)


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


def create_task_details_preemptive(
    solution: MS_RCPSPSolution_Preemptive_Variant, time_to_cut: int
):
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


def create_task_details_classic(solution: MS_RCPSPSolution_Variant, time_to_cut: int):
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


def sgs_debug_preemptive(model=None):
    if model is None:
        # model = create_toy_msrcpsp_variant()
        # files = get_data_available()
        files = [f for f in get_data_available() if "100_5_22_15.def" in f]
        model_msrcpsp, new_tame_to_original_task_id = parse_file(
            files[0], max_horizon=2000, one_unit_per_task=False, preemptive=True
        )
        model = model_msrcpsp.to_variant_model()

    dummy_solution = model.get_dummy_solution()
    print("dummy_solution.priority_list_task: ", dummy_solution.priority_list_task)
    print(
        "dummy_solution.priority_worker_per_task: ",
        dummy_solution.priority_worker_per_task,
    )
    print("dummy_solution.modes_vector: ", dummy_solution.modes_vector)
    print(model.evaluate(dummy_solution))
    print(model.satisfy(dummy_solution))
    timesgs2 = int(dummy_solution.get_end_time(model.sink_task) / 2)
    completed, ongoing = create_task_details_preemptive(dummy_solution, timesgs2)
    import time

    from discrete_optimization.rcpsp_multiskill.plots.plot_solution import (
        plot_resource_individual_gantt_preemptive,
        plt,
    )

    for i in range(10):
        t = time.time()
        dummy_solution.do_recompute(fast=False)
        t_end = time.time()
        print("Full SGS ", t_end - t, " seconds slow")
        print("Evaluate : ", model.evaluate(dummy_solution))
        t = time.time()
        dummy_solution.do_recompute(fast=True)
        t_end = time.time()
        print("Full SGS ", t_end - t, " seconds fast")
        print("Evaluate : ", model.evaluate(dummy_solution))
    plot_resource_individual_gantt_preemptive(
        rcpsp_model=model, rcpsp_sol=dummy_solution
    )
    plt.show()

    for i in range(100):
        t = time.time()
        dummy_solution.run_sgs_partial(
            current_t=timesgs2,
            completed_tasks=completed,
            scheduled_tasks_start_times=ongoing,
            fast=False,
        )
        t_end = time.time()
        print(t_end - t, " seconds slow")
        t = time.time()
        dummy_solution.run_sgs_partial(
            current_t=timesgs2,
            completed_tasks=completed,
            scheduled_tasks_start_times=ongoing,
            fast=True,
        )
        t_end = time.time()
        print(t_end - t, " seconds fast")
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


def sgs_debug_om():
    from scheduler_om_pi2.parsing.cn_file_parser import (
        RepoVersion,
        build_multiskill,
        parse,
    )

    model: MS_RCPSPModel = build_multiskill(
        index=0,
        include_zones=False,
        cut_too_long_task=False,
        remove_masked=True,
        add_missing_worker_super_skilled=0 == 0,
        remove_resources_type=True,
        add_varying_ressource=True,
        preemptive=True,
        repo_version=RepoVersion.DO,
    )
    model = model.to_variant_model()
    solver = LS_RCPSP_Solver(model=model, ls_solver=LS_SOLVER.SA)
    result = solver.solve(nb_iteration_max=500, init_solution_process=False)

    # sgs_debug_preemptive(model)


if __name__ == "__main__":
    sgs_debug_preemptive()