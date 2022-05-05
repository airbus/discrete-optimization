from discrete_optimization.rcpsp.solver.rcpsp_cp_lns_solver import (
    LNS_CP_RCPSP_SOLVER,
    OptionNeighbor,
)
from discrete_optimization.rcpsp_multiskill.multiskill_to_rcpsp import MultiSkillToRCPSP


def run_ms_to_rcpsp_imopse():
    import os
    import random

    from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill_parser import (
        get_data_available,
        parse_file,
    )

    files = get_data_available()
    files = [f for f in get_data_available() if "200_40_133_15.def" in f]
    random.shuffle(files)
    model_msrcpsp, new_name_to_original_task_id = parse_file(files[0], max_horizon=2000)
    print(model_msrcpsp)

    algorithm = MultiSkillToRCPSP(model_msrcpsp)
    rcpsp_model = algorithm.construct_rcpsp_by_worker_type(
        limit_number_of_mode_per_task=False,
        check_resource_compliance=True,
        max_number_of_mode=100,
    )
    print(rcpsp_model)
    print("Hola")


def solve_rcpsp_imopse():
    import os
    import random

    from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill_parser import (
        get_data_available,
        parse_file,
    )

    files = get_data_available()
    files = [f for f in get_data_available() if "200_40_133_15.def" in f]
    random.shuffle(files)
    model_msrcpsp, new_name_to_original_task_id = parse_file(files[0], max_horizon=2000)
    print(model_msrcpsp)
    algorithm = MultiSkillToRCPSP(model_msrcpsp)
    rcpsp_model = algorithm.construct_rcpsp_by_worker_type(
        limit_number_of_mode_per_task=True,
        check_resource_compliance=True,
        max_number_of_mode=5,
        one_worker_type_per_task=True,
    )
    from discrete_optimization.rcpsp.solver.cp_solvers import (
        CP_MRCPSP_MZN,
        CPSolverName,
        ParametersCP,
    )

    params_cp = ParametersCP.default()
    params_cp.TimeLimit = 200
    params_cp.TimeLimit_iter0 = 300
    if True:
        params_cp.free_search = False
        lns_solver = LNS_CP_RCPSP_SOLVER(
            rcpsp_model=rcpsp_model, option_neighbor=OptionNeighbor.MIX_FAST
        )
        result_storage = lns_solver.solve(
            parameters_cp=params_cp,
            nb_iteration_lns=20,
            max_time_seconds=2000,
            nb_iteration_no_improvement=200,
            skip_first_iteration=False,
        )
    else:
        solver = CP_MRCPSP_MZN(
            rcpsp_model=rcpsp_model, cp_solver_name=CPSolverName.CHUFFED
        )
        solver.init_model(output_type=True)
        params_cp = ParametersCP.default()
        params_cp.TimeLimit = 1000
        result_storage = solver.solve(parameters_cp=params_cp)
    from discrete_optimization.rcpsp.rcpsp_utils import (
        plot_resource_individual_gantt,
        plot_ressource_view,
        plot_task_gantt,
        plt,
    )

    best_solution = result_storage.get_best_solution()
    plot_ressource_view(rcpsp_model=rcpsp_model, rcpsp_sol=best_solution)
    plot_task_gantt(rcpsp_model=rcpsp_model, rcpsp_sol=best_solution)
    plot_resource_individual_gantt(rcpsp_model=rcpsp_model, rcpsp_sol=best_solution)
    plt.show()
    print(rcpsp_model)
    print("Hola")


if __name__ == "__main__":
    solve_rcpsp_imopse()
