from discrete_optimization.generic_tools.cp_tools import CPSolverName, ParametersCP
from discrete_optimization.rcpsp.rcpsp_model_preemptive import (
    RCPSPModelPreemptive,
    RCPSPSolutionPreemptive,
    get_rcpsp_modelp_preemptive,
)
from discrete_optimization.rcpsp.rcpsp_parser import (
    RCPSPModel,
    get_data_available,
    parse_file,
)
from discrete_optimization.rcpsp.solver.cp_solvers import (
    CP_MRCPSP_MZN_PREEMMPTIVE,
    CP_RCPSP_MZN_PREEMMPTIVE,
)


def run_preemptive():
    resource_r1 = []
    for i in range(40):
        resource_r1 += [3, 3, 3, 0, 0]
    rcpsp_problem = RCPSPModelPreemptive(
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
    files = get_data_available()
    files = [f for f in files if "j1201_1.sm" in f]  # Single mode RCPSP
    file_path = files[0]
    do_psp_lib = True
    if do_psp_lib:
        rcpsp_model: RCPSPModelPreemptive = parse_file(file_path)
        rcpsp_problem = RCPSPModelPreemptive(
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

    solution = rcpsp_problem.get_dummy_solution()
    import time

    for i in range(100):
        t = time.time()
        solution.generate_schedule_from_permutation_serial_sgs(do_fast=False)
        print(time.time() - t, "not fast")
        print(rcpsp_problem.evaluate(solution))
        t = time.time()
        solution.generate_schedule_from_permutation_serial_sgs(do_fast=True)
        print(time.time() - t, "fast")
        print(rcpsp_problem.evaluate(solution))

    from copy import deepcopy

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

    for i in range(100):
        t = time.time()
        solution.generate_schedule_from_permutation_serial_sgs_2(
            current_t=timesgs2,
            partial_schedule=partial_schedule,
            completed_tasks=completed,
            do_fast=False,
        )
        print(time.time() - t, "not fast")
        print(rcpsp_problem.evaluate(solution))
        t = time.time()
        solution.generate_schedule_from_permutation_serial_sgs_2(
            current_t=timesgs2,
            partial_schedule=partial_schedule,
            completed_tasks=completed,
            do_fast=True,
        )
        print(time.time() - t, "fast")
        print(rcpsp_problem.evaluate(solution))

    for o in solution.rcpsp_schedule:
        print("Compare previous ", o, solution.rcpsp_schedule[o], previous_schedule[o])

    for o in partial_schedule:
        print(solution.rcpsp_schedule[o], partial_schedule[o])
    for o in completed:
        print("completed")
        print(solution.rcpsp_schedule[o], partial_schedule[o])
    print(rcpsp_problem.satisfy(solution))
    print(rcpsp_problem.evaluate(solution))


def preemptive_cp():
    from discrete_optimization.rcpsp.rcpsp_model_preemptive import RCPSPModelPreemptive

    resource_r1 = []
    for i in range(40):
        resource_r1 += [3, 3, 3, 0, 0]
    rcpsp_problem = RCPSPModelPreemptive(
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

    solver = CP_RCPSP_MZN_PREEMMPTIVE(
        rcpsp_model=rcpsp_problem, cp_solver_name=CPSolverName.CHUFFED
    )
    solver.init_model(
        output_type=True,
        model_type="single-preemptive-calendar",
        nb_preemptive=7,
        possibly_preemptive=[True for k in rcpsp_problem.tasks_list],
        max_preempted=20,
    )
    parameters_cp = ParametersCP.default()
    parameters_cp.TimeLimit = 100
    result_store = solver.solve(parameters_cp=parameters_cp)
    best_solution = result_store.get_best_solution()

    print(rcpsp_problem.satisfy(best_solution))
    print(rcpsp_problem.evaluate(best_solution))
    print(result_store)
    from discrete_optimization.rcpsp.plots.rcpsp_utils_preemptive import (
        plot_resource_individual_gantt,
        plot_ressource_view,
        plot_task_gantt,
        plt,
    )

    plot_ressource_view(rcpsp_problem, best_solution)
    plot_resource_individual_gantt(rcpsp_problem, best_solution)
    plot_task_gantt(rcpsp_problem, best_solution)
    plt.show()


def preemptive_multimode_cp():
    from discrete_optimization.rcpsp.rcpsp_model_preemptive import RCPSPModelPreemptive

    resource_r1 = []
    for i in range(40):
        resource_r1 += [3, 3, 3, 0, 0]
    rcpsp_problem = RCPSPModelPreemptive(
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

    solver = CP_MRCPSP_MZN_PREEMMPTIVE(
        rcpsp_model=rcpsp_problem, cp_solver_name=CPSolverName.CHUFFED
    )
    solver.init_model(
        output_type=True,
        model_type="multi-preemptive-calendar",
        nb_preemptive=7,
        possibly_preemptive=[True for k in rcpsp_problem.tasks_list],
        max_preempted=20,
    )
    parameters_cp = ParametersCP.default()
    parameters_cp.TimeLimit = 100
    result_store = solver.solve(parameters_cp=parameters_cp)
    best_solution = result_store.get_best_solution()

    print(rcpsp_problem.satisfy(best_solution))
    print(rcpsp_problem.evaluate(best_solution))
    print(result_store)
    from discrete_optimization.rcpsp.plots.rcpsp_utils_preemptive import (
        plot_resource_individual_gantt,
        plot_ressource_view,
        plot_task_gantt,
        plt,
    )

    plot_ressource_view(rcpsp_problem, best_solution)
    plot_resource_individual_gantt(rcpsp_problem, best_solution)
    plot_task_gantt(rcpsp_problem, best_solution)
    plt.show()


def preemptive_psplib():
    files = get_data_available()
    files = [f for f in files if "j601_5.sm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_model: RCPSPModel = parse_file(file_path)
    rcpsp_problem = get_rcpsp_modelp_preemptive(rcpsp_model)
    solver = CP_RCPSP_MZN_PREEMMPTIVE(
        rcpsp_model=rcpsp_problem, cp_solver_name=CPSolverName.CHUFFED
    )
    solver.init_model(
        output_type=True,
        model_type="single-preemptive",
        possibly_preemptive=[True for t in rcpsp_problem.tasks_list],
        nb_preemptive=4,
        max_preempted=20,
    )
    parameters_cp = ParametersCP.default()
    parameters_cp.TimeLimit = 100
    result_store = solver.solve(parameters_cp=parameters_cp)
    best_solution = result_store.get_best_solution()
    print(rcpsp_problem.satisfy(best_solution))
    print(rcpsp_problem.evaluate(best_solution))
    print(result_store)
    from discrete_optimization.rcpsp.plots.rcpsp_utils_preemptive import (
        plot_resource_individual_gantt,
        plot_ressource_view,
        plot_task_gantt,
        plt,
    )

    plot_ressource_view(rcpsp_problem, best_solution)
    plot_resource_individual_gantt(rcpsp_problem, best_solution)
    plot_task_gantt(rcpsp_problem, best_solution)
    plt.show()


def preemptive_multimode_psplib():
    files = get_data_available()
    files = [f for f in files if "j1010_5.mm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_model: RCPSPModelPreemptive = parse_file(file_path)
    rcpsp_problem = get_rcpsp_modelp_preemptive(rcpsp_model)
    solver = CP_MRCPSP_MZN_PREEMMPTIVE(
        rcpsp_model=rcpsp_problem, cp_solver_name=CPSolverName.CHUFFED
    )
    solver.init_model(
        output_type=True,
        model_type="multi-preemptive",
        possibly_preemptive=[True for t in rcpsp_problem.tasks_list],
        nb_preemptive=4,
        max_preempted=20,
    )
    parameters_cp = ParametersCP.default()
    parameters_cp.TimeLimit = 100
    result_store = solver.solve(parameters_cp=parameters_cp)
    best_solution = result_store.get_best_solution()
    print(rcpsp_problem.satisfy(best_solution))
    print(rcpsp_problem.evaluate(best_solution))
    print(result_store)
    from discrete_optimization.rcpsp.plots.rcpsp_utils_preemptive import (
        plot_resource_individual_gantt,
        plot_ressource_view,
        plot_task_gantt,
        plt,
    )

    plot_ressource_view(rcpsp_problem, best_solution)
    plot_resource_individual_gantt(rcpsp_problem, best_solution)
    plot_task_gantt(rcpsp_problem, best_solution)
    plt.show()


def preeemptive_sgs():
    files = get_data_available()
    files = [f for f in files if "j601_5.sm" in f]  # Single mode RCPSP
    file_path = files[0]
    rcpsp_model: RCPSPModel = parse_file(file_path)
    rcpsp_problem = get_rcpsp_modelp_preemptive(rcpsp_model)
    solution: RCPSPSolutionPreemptive = rcpsp_problem.get_dummy_solution()
    rcpsp_problem.duration_subtask = {t: (True, 2) for t in rcpsp_problem.tasks_list}
    rcpsp_problem.any_duration_subtask_limited = True
    rcpsp_problem.update_function()
    solution = rcpsp_problem.get_dummy_solution()
    print("Max number of preemption :", solution.get_max_preempted())
    print("Nb task preempted : ", solution.get_nb_task_preemption())
    print("Makespan : ", solution.get_end_time(rcpsp_problem.sink_task))


if __name__ == "__main__":
    preeemptive_sgs()
