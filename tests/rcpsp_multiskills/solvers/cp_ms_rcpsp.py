from typing import Dict, List, Set

import matplotlib.pyplot as plt
from discrete_optimization.datasets import get_data_home
from discrete_optimization.generic_tools.cp_tools import CPSolverName
from discrete_optimization.generic_tools.lp_tools import MilpSolverName, ParametersMilp
from discrete_optimization.rcpsp.rcpsp_model import RCPSPSolution
from discrete_optimization.rcpsp.rcpsp_utils import plot_ressource_view
from discrete_optimization.rcpsp.solver.rcpsp_lp_lns_solver import InitialMethodRCPSP
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import (
    Employee,
    MS_RCPSPModel,
    MS_RCPSPModel_Variant,
    MS_RCPSPSolution,
    SkillDetail,
)
from discrete_optimization.rcpsp_multiskill.solvers.cp_solvers import (
    CP_MS_MRCPSP_MZN,
    ParametersCP,
)
from discrete_optimization.rcpsp_multiskill.solvers.lp_model import LP_Solver_MRSCPSP
from discrete_optimization.rcpsp_multiskill.solvers.ms_rcpsp_cp_lns_solver import (
    LNS_CP_MS_RCPSP_SOLVER,
    OptionNeighbor,
)
from discrete_optimization.rcpsp_multiskill.solvers.ms_rcpsp_lp_lns_solver import (
    InitialSolutionMS_RCPSP,
)

# from tests.rcpsp_multiskills.solvers.instance_creator import create_ms_rcpsp_demo


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


def create_toy_santi():
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


def cp_toy_model():
    model_msrcpsp = create_toy_santi()

    cp_model = CP_MS_MRCPSP_MZN(
        rcpsp_model=model_msrcpsp, cp_solver_name=CPSolverName.GECODE
    )
    cp_model.init_model(
        add_calendar_constraint_unit=False,
        fake_tasks=True,
        output_type=True,
        exact_skills_need=False,
    )
    parameters_cp = ParametersCP.default()
    parameters_cp.intermediate_solution = True
    parameters_cp.TimeLimit = 2000
    result_storage = cp_model.solve(parameters_cp=parameters_cp)
    solution: MS_RCPSPSolution = result_storage.get_best_solution()
    from discrete_optimization.rcpsp_multiskill.plots.plot_solution import (
        plot_resource_individual_gantt,
        plot_task_gantt,
        plt,
    )

    plot_resource_individual_gantt(rcpsp_model=model_msrcpsp, rcpsp_sol=solution)
    plot_task_gantt(rcpsp_model=model_msrcpsp, rcpsp_sol=solution)
    plt.show()


def cp_imopse():
    import os

    from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill_parser import (
        folder_to_do_solution,
        get_data_available,
        parse_file,
        write_solution,
    )

    file = [f for f in get_data_available() if "100_5_20_9_D3.def" in f][0]
    model_msrcpsp, new_tame_to_original_task_id = parse_file(file, max_horizon=1000)
    cp_model = CP_MS_MRCPSP_MZN(
        rcpsp_model=model_msrcpsp,
        one_ressource_per_task=True,
        cp_solver_name=CPSolverName.CHUFFED,
    )
    cp_model.init_model(
        add_calendar_constraint_unit=False,
        fake_tasks=False,
        add_objective_makespan=True,
        ignore_sec_objective=True,
        output_type=True,
        max_time=1000,
        # here you put makespan constraint. by default, would use model_msrcpsp.horizon if not provided.
        exact_skills_need=False,
    )
    parameters_cp = ParametersCP.default()
    parameters_cp.free_search = True
    # With parameters_cp.free_search=True you get fast results but quite bad !
    # but can be a good thing to be used in findmus algo
    # with free_search=False, you get first results after 10 seconds or so, but good quality.
    parameters_cp.intermediate_solution = True
    parameters_cp.TimeLimit = 2000
    result_storage = cp_model.solve(parameters_cp=parameters_cp)
    solution: MS_RCPSPSolution = result_storage.get_best_solution()
    if False:
        write_solution(
            solution=solution,
            new_tame_to_original_task_id=new_tame_to_original_task_id,
            file_path=os.path.join(
                folder_to_do_solution, os.path.basename(file) + ".sol"
            ),
        )
        print(model_msrcpsp.evaluate(solution))
        print("Satisfy : ", model_msrcpsp.satisfy(solution))
        rebuilt_sol_rcpsp = RCPSPSolution(
            problem=model_rcpsp,
            rcpsp_permutation=None,
            rcpsp_schedule=solution.schedule,
            rcpsp_modes=[solution.modes[x] for x in range(2, model_rcpsp.n_jobs + 2)],
        )
        plot_ressource_view(model_rcpsp, rebuilt_sol_rcpsp)
        plt.show()


def only_cp_imopse():
    import os
    import random

    from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill_parser import (
        get_data_available,
        get_results_available,
        parse_file,
        write_solution,
    )

    folder_to_do_only_cp_solution = (
        f"{get_data_home()}/rcpsp_multiskill/do_cp_solutions"
    )
    file = [f for f in get_data_available() if "100_10_47_9.def" in f][0]
    # print(files[1])
    files = [f for f in get_data_available() if "100_20_46_15.def" in f]
    files = get_data_available()
    random.shuffle(files)
    files = get_data_available()
    files = [f for f in get_data_available() if "100_20_46_15.def" in f]
    files = get_data_available()
    # files = [f for f in get_data_available() if '100_20_47_9.def' in f]
    random.shuffle(files)

    for file in files:
        if any(
            os.path.basename(file) in f
            for f in get_results_available(folder_to_do_only_cp_solution)
        ):
            print("Already done")
            continue
        model_msrcpsp, new_tame_to_original_task_id = parse_file(file, max_horizon=2000)
        # if model_msrcpsp.nb_tasks >= 185:
        #     continue
        initial_solution_provider = InitialSolutionMS_RCPSP(
            problem=model_msrcpsp,
            initial_method=InitialMethodRCPSP.PILE_CALENDAR,
            params_objective_function=None,
        )
        solution = initial_solution_provider.get_starting_solution().get_best_solution()
        makespan = model_msrcpsp.evaluate(solution)["makespan"]
        model_msrcpsp.horizon = makespan + 5
        print(file)
        print(model_msrcpsp.horizon)
        model_rcpsp = model_msrcpsp.build_multimode_rcpsp_calendar_representative()
        cp_solver = CP_MS_MRCPSP_MZN(
            rcpsp_model=model_msrcpsp,
            cp_solver_name=CPSolverName.CHUFFED,
            one_ressource_per_task=True,
        )
        parameters_cp = ParametersCP.default()
        parameters_cp.intermediate_solution = True
        parameters_cp.all_solutions = False
        parameters_cp.TimeLimit = 6000
        parameters_cp.TimeLimit_iter0 = 60
        result_storage = cp_solver.solve(parameters_cp)
        solution: MS_RCPSPSolution = result_storage.get_best_solution()
        # for task in solution.employee_usage:
        #     print(task, len(solution.employee_usage[task]))
        if solution is None:
            continue
        write_solution(
            solution=solution,
            new_tame_to_original_task_id=new_tame_to_original_task_id,
            file_path=os.path.join(
                folder_to_do_only_cp_solution, os.path.basename(file) + ".sol"
            ),
        )
        print(model_msrcpsp.evaluate(solution))
        print("Satisfy : ", model_msrcpsp.satisfy(solution))
        # rebuilt_sol_rcpsp = RCPSPSolution(problem=model_rcpsp,
        #                                   rcpsp_permutation=None,
        #                                   rcpsp_schedule=solution.schedule,
        #                                   rcpsp_modes=[solution.modes[x]
        #                                                for x in range(2, model_rcpsp.n_jobs + 2)])
        # plot_ressource_view(model_rcpsp, rebuilt_sol_rcpsp)
        # plt.show()


def lns_cp_imopse():
    import os
    import random

    from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill_parser import (
        get_data_available,
        get_results_available,
        parse_file,
        write_solution,
    )

    folder_to_do_solution = f"{get_data_home()}/rcpsp_multiskill/do_solutions"
    files = get_data_available()
    random.shuffle(files)
    to_rerun = [
        f[:-4]
        for f in [
            "100_5_22_15.def.sol",
            "100_10_26_15.def.sol",
            "100_5_64_9.def.sol",
            "100_10_65_15.def.sol",
            "100_5_64_15.def.sol",
        ]
    ]
    for file in files:
        print(os.path.basename(file))
        if any(
            os.path.basename(file) in f and os.path.basename(file) not in to_rerun
            for f in get_results_available(folder_to_do_solution)
        ):
            print("Already done")
            continue
        model_msrcpsp, new_tame_to_original_task_id = parse_file(file, max_horizon=2000)
        # if model_msrcpsp.nb_tasks >= 185:
        #     continue
        initial_solution_provider = InitialSolutionMS_RCPSP(
            problem=model_msrcpsp,
            initial_method=InitialMethodRCPSP.PILE_CALENDAR,
            params_objective_function=None,
        )
        solution = initial_solution_provider.get_starting_solution().get_best_solution()
        makespan = model_msrcpsp.evaluate(solution)["makespan"]
        model_msrcpsp.horizon = makespan + 5
        model_rcpsp = model_msrcpsp.build_multimode_rcpsp_calendar_representative()
        lns_cp = LNS_CP_MS_RCPSP_SOLVER(
            rcpsp_model=model_msrcpsp,
            option_neighbor=OptionNeighbor.MIX_FAST
            if model_msrcpsp.nb_tasks >= 190
            else OptionNeighbor.MIX_ALL,
            one_ressource_per_task=True,
        )
        parameters_cp = ParametersCP.default()
        parameters_cp.intermediate_solution = True
        parameters_cp.all_solutions = False
        parameters_cp.TimeLimit = 100
        parameters_cp.TimeLimit_iter0 = 60
        result_storage = lns_cp.solve(
            parameters_cp=parameters_cp,
            nb_iteration_lns=3000,
            max_time_seconds=7200,
            nb_iteration_no_improvement=500,
            skip_first_iteration=False,
        )
        solution: MS_RCPSPSolution = result_storage.get_best_solution()
        # for task in solution.employee_usage:
        #     print(task, len(solution.employee_usage[task]))
        write_solution(
            solution=solution,
            new_tame_to_original_task_id=new_tame_to_original_task_id,
            file_path=os.path.join(
                folder_to_do_solution, os.path.basename(file) + ".sol"
            ),
        )
        print(model_msrcpsp.evaluate(solution))
        print("Satisfy : ", model_msrcpsp.satisfy(solution))
        # rebuilt_sol_rcpsp = RCPSPSolution(problem=model_rcpsp,
        #                                   rcpsp_permutation=None,
        #                                   rcpsp_schedule=solution.schedule,
        #                                   rcpsp_modes=[solution.modes[x]
        #                                                for x in range(2, model_rcpsp.n_jobs + 2)])
        # plot_ressource_view(model_rcpsp, rebuilt_sol_rcpsp)
        # plt.show()


def lns_small_neighbor():
    import os
    import random

    from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill_parser import (
        get_data_available,
        parse_file,
        write_solution,
    )

    # file = [f for f in get_data_available() if "100_5_64_15.def.sol" in f][0]
    # print(files[1])
    files = get_data_available()
    # files = [f for f in get_data_available() if '100_10_65_15.def' in f]
    random.shuffle(files)
    folder_do_small_solution = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../results/do_cp_largeneighbor_test/",
    )
    if not os.path.exists(folder_do_small_solution):
        os.makedirs(folder_do_small_solution)
    files = [f for f in get_data_available() if "100_5_64_9.def" in f]
    for file in files:
        print(os.path.basename(file))
        if any(
            os.path.basename(file) in f for f in os.listdir(folder_do_small_solution)
        ):
            print("Already done")
        model_msrcpsp, new_tame_to_original_task_id = parse_file(file, max_horizon=2000)
        # if model_msrcpsp.nb_tasks >= 185:
        #     continue
        initial_solution_provider = InitialSolutionMS_RCPSP(
            problem=model_msrcpsp,
            initial_method=InitialMethodRCPSP.PILE_CALENDAR,
            params_objective_function=None,
        )
        solution = initial_solution_provider.get_starting_solution().get_best_solution()
        makespan = model_msrcpsp.evaluate(solution)["makespan"]
        model_msrcpsp.horizon = makespan + 5
        model_rcpsp = model_msrcpsp.build_multimode_rcpsp_calendar_representative()
        lns_cp = LNS_CP_MS_RCPSP_SOLVER(
            rcpsp_model=model_msrcpsp,
            option_neighbor=OptionNeighbor.MIX_LARGE_NEIGH,
            one_ressource_per_task=True,
        )
        parameters_cp = ParametersCP.default()
        parameters_cp.intermediate_solution = True
        parameters_cp.all_solutions = False
        parameters_cp.TimeLimit = 200
        parameters_cp.TimeLimit_iter0 = 60
        result_storage = lns_cp.solve(
            parameters_cp=parameters_cp,
            nb_iteration_lns=3000,
            max_time_seconds=10000,
            nb_iteration_no_improvement=1000,
            skip_first_iteration=False,
        )
        solution: MS_RCPSPSolution = result_storage.get_best_solution()
        # for task in solution.employee_usage:
        #     print(task, len(solution.employee_usage[task]))
        write_solution(
            solution=solution,
            new_tame_to_original_task_id=new_tame_to_original_task_id,
            file_path=os.path.join(
                folder_do_small_solution, os.path.basename(file) + ".sol"
            ),
        )
        print(model_msrcpsp.evaluate(solution))
        print("Satisfy : ", model_msrcpsp.satisfy(solution))
        # rebuilt_sol_rcpsp = RCPSPSolution(problem=model_rcpsp,
        #                                   rcpsp_permutation=None,
        #                                   rcpsp_schedule=solution.schedule,
        #                                   rcpsp_modes=[solution.modes[x]
        #                                                for x in range(2, model_rcpsp.n_jobs + 2)])
        # plot_ressource_view(model_rcpsp, rebuilt_sol_rcpsp)
        # plt.show()


def lns_example():
    from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill_parser import (
        folder_to_do_solution,
        get_data_available,
        parse_file,
        write_solution,
    )

    file = [f for f in get_data_available() if "100_5_22_15.def" in f][0]
    # file = [f for f in get_data_available() if "100_5_20_9_D3.def" in f][0]

    model_msrcpsp, new_tame_to_original_task_id = parse_file(file, max_horizon=2000)
    model_msrcpsp = model_msrcpsp.to_variant_model()
    initial_solution_provider = InitialSolutionMS_RCPSP(
        problem=model_msrcpsp,
        initial_method=InitialMethodRCPSP.PILE_CALENDAR,
        params_objective_function=None,
    )
    solution = initial_solution_provider.get_starting_solution().get_best_solution()
    makespan = model_msrcpsp.evaluate(solution)["makespan"]
    model_msrcpsp.horizon = makespan + 5
    model_rcpsp = model_msrcpsp.build_multimode_rcpsp_calendar_representative()
    lns_cp = LNS_CP_MS_RCPSP_SOLVER(
        rcpsp_model=model_msrcpsp,
        option_neighbor=OptionNeighbor.MIX_ALL,
        one_ressource_per_task=True,
        fake_tasks=True,
        output_type=True,
        exact_skills_need=False,
    )
    parameters_cp = ParametersCP.default()
    parameters_cp.intermediate_solution = True
    parameters_cp.all_solutions = False
    parameters_cp.TimeLimit = 100
    parameters_cp.TimeLimit_iter0 = 60
    result_storage = lns_cp.solve(
        parameters_cp=parameters_cp,
        nb_iteration_lns=3000,
        max_time_seconds=7200,
        nb_iteration_no_improvement=500,
        skip_first_iteration=False,
    )
    solution: MS_RCPSPSolution = result_storage.get_best_solution()


if __name__ == "__main__":
    cp_imopse()
