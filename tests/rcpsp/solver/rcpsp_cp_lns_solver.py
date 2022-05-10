import matplotlib.pyplot as plt
from discrete_optimization.generic_tools.cp_tools import CPSolverName, ParametersCP
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveHandling,
    ParamsObjectiveFunction,
    get_default_objective_setup,
)
from discrete_optimization.generic_tools.lns_cp import LNS_CP, InitialSolution
from discrete_optimization.rcpsp.rcpsp_model import (
    MultiModeRCPSPModel,
    RCPSPModel,
    RCPSPSolution,
    SingleModeRCPSPModel,
)
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file
from discrete_optimization.rcpsp.rcpsp_utils import (
    plot_resource_individual_gantt,
    plot_ressource_view,
)
from discrete_optimization.rcpsp.solver.cp_solvers import CP_MRCPSP_MZN, CP_RCPSP_MZN
from discrete_optimization.rcpsp.solver.rcpsp_cp_lns_solver import (
    LNS_CP_RCPSP_SOLVER,
    ConstraintHandlerStartTimeInterval_CP,
    OptionNeighbor,
)
from discrete_optimization.rcpsp.solver.rcpsp_lp_lns_solver import (
    LNS_LP_RCPSP_SOLVER,
    InitialMethodRCPSP,
    InitialSolutionRCPSP,
)


def lns_single_mode():
    files_available = get_data_available()
    file = [f for f in files_available if "j1201_1.sm" in f][0]
    rcpsp_problem: RCPSPModel = parse_file(file)
    solver = CP_RCPSP_MZN(
        rcpsp_model=rcpsp_problem, cp_solver_name=CPSolverName.CHUFFED
    )
    solver.init_model(output_type=True)
    parameters_cp = ParametersCP.default()
    params_objective_function = get_default_objective_setup(problem=rcpsp_problem)
    constraint_handler = ConstraintHandlerStartTimeInterval_CP(
        problem=rcpsp_problem,
        fraction_to_fix=1.0,
        # here i want to apply bounds constraint on all the tasks
        minus_delta=10,
        plus_delta=10,
    )
    from discrete_optimization.generic_tools.lns_mip import TrivialInitialSolution
    from discrete_optimization.generic_tools.result_storage.result_storage import (
        from_solutions_to_result_storage,
    )

    some_solution = rcpsp_problem.get_dummy_solution()  # starting solution
    initial_solution_provider = TrivialInitialSolution(
        solution=from_solutions_to_result_storage(
            [some_solution], problem=rcpsp_problem
        )
    )
    # initial_solution_provider = InitialSolutionRCPSP(problem=rcpsp_problem,
    #                                                  initial_method=InitialMethodRCPSP.PILE,
    #                                                  params_objective_function=params_objective_function)
    lns_solver = LNS_CP(
        problem=rcpsp_problem,
        cp_solver=solver,
        initial_solution_provider=initial_solution_provider,
        constraint_handler=constraint_handler,
        params_objective_function=params_objective_function,
    )
    result_store = lns_solver.solve_lns(
        parameters_cp=parameters_cp, nb_iteration_lns=300
    )
    solution, fit = result_store.get_best_solution_fit()
    solution_rebuilt = RCPSPSolution(
        problem=rcpsp_problem, rcpsp_permutation=solution.rcpsp_permutation
    )
    fit_2 = rcpsp_problem.evaluate(solution_rebuilt)
    print(rcpsp_problem.evaluate(solution), fit_2)
    print("Satisfy : ", rcpsp_problem.satisfy(solution))
    plot_resource_individual_gantt(rcpsp_problem, solution)
    plot_ressource_view(rcpsp_problem, solution)
    plt.show()


def lns_multi_mode():
    files_available = get_data_available()
    file = [f for f in files_available if "j1010_9.mm" in f][0]
    rcpsp_problem: RCPSPModel = parse_file(file)
    if isinstance(rcpsp_problem, MultiModeRCPSPModel):
        rcpsp_problem.set_fixed_modes([1 for i in range(rcpsp_problem.n_jobs)])
    solver = CP_MRCPSP_MZN(
        rcpsp_model=rcpsp_problem, cp_solver_name=CPSolverName.CHUFFED
    )
    solver.init_model()
    parameters_cp = ParametersCP.default()
    params_objective_function = get_default_objective_setup(problem=rcpsp_problem)
    # constraint_handler = ConstraintHandlerFixStartTime(problem=rcpsp_problem,
    #                                                    fraction_fix_start_time=0.5)
    constraint_handler = ConstraintHandlerStartTimeInterval_CP(
        problem=rcpsp_problem, fraction_to_fix=0.7, minus_delta=5, plus_delta=5
    )
    initial_solution_provider = InitialSolutionRCPSP(
        problem=rcpsp_problem,
        initial_method=InitialMethodRCPSP.LS,
        params_objective_function=params_objective_function,
    )
    lns_solver = LNS_CP(
        problem=rcpsp_problem,
        cp_solver=solver,
        initial_solution_provider=initial_solution_provider,
        constraint_handler=constraint_handler,
        params_objective_function=params_objective_function,
    )
    result_store = lns_solver.solve_lns(
        parameters_cp=parameters_cp, nb_iteration_lns=300
    )
    solution, fit = result_store.get_best_solution_fit()
    solution_rebuilt = RCPSPSolution(
        problem=rcpsp_problem,
        rcpsp_permutation=solution.rcpsp_permutation,
        rcpsp_modes=solution.rcpsp_modes,
    )
    fit_2 = rcpsp_problem.evaluate(solution_rebuilt)
    print(rcpsp_problem.evaluate(solution), fit_2)
    print("Satisfy : ", rcpsp_problem.satisfy(solution))
    plot_resource_individual_gantt(rcpsp_problem, solution)
    plot_ressource_view(rcpsp_problem, solution)
    plt.show()


def lns_solver():
    files_available = get_data_available()
    file = [f for f in files_available if "j1201_1.sm" in f][0]
    rcpsp_problem: SingleModeRCPSPModel = parse_file(file)
    parameters_cp = ParametersCP.default()
    lns_solver = LNS_CP_RCPSP_SOLVER(
        rcpsp_model=rcpsp_problem, option_neighbor=OptionNeighbor.MIX_ALL
    )
    result_store = lns_solver.solve(
        parameters_cp=parameters_cp,
        nb_iteration_lns=20,
        max_time_seconds=60,
        nb_iteration_no_improvement=200,
    )
    solution, fit = result_store.get_best_solution_fit()
    solution_rebuilt = RCPSPSolution(
        problem=rcpsp_problem, rcpsp_permutation=solution.rcpsp_permutation
    )
    fit_2 = rcpsp_problem.evaluate(solution_rebuilt)
    print(rcpsp_problem.evaluate(solution), fit_2)
    print("Satisfy : ", rcpsp_problem.satisfy(solution))
    fig, ax = plt.subplots()
    ax.plot([x[1] for x in result_store.list_solution_fits])
    plot_resource_individual_gantt(rcpsp_problem, solution)
    plot_ressource_view(rcpsp_problem, solution)
    plt.show()


if __name__ == "__main__":
    lns_single_mode()
