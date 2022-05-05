import matplotlib.pyplot as plt
from discrete_optimization.generic_tools.lp_tools import ParametersMilp
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.rcpsp_data_generator import (
    generate_rcpsp_with_helper_tasks_data,
)
from discrete_optimization.rcpsp.rcpsp_model import (
    MethodBaseRobustification,
    MethodRobustification,
    MultiModeRCPSPModel,
    PartialSolution,
    RCPSPModel,
    RCPSPSolution,
    SingleModeRCPSPModel,
    UncertainRCPSPModel,
    create_poisson_laws_duration,
)
from discrete_optimization.rcpsp.rcpsp_parser import (
    files_available,
    get_data_available,
    parse_file,
)
from discrete_optimization.rcpsp.rcpsp_utils import (
    kendall_tau_similarity,
    plot_resource_individual_gantt,
    plot_ressource_view,
)
from discrete_optimization.rcpsp.solver.rcpsp_lp_solver import (
    LP_MRCPSP,
    LP_RCPSP,
    LP_RCPSP_Solver,
)


def single_mode_rcpsp_lp_cbc():
    file = [f for f in files_available if "j301_1.sm" in f][0]
    rcpsp_problem: SingleModeRCPSPModel = parse_file(file)
    solver = LP_RCPSP(rcpsp_model=rcpsp_problem, lp_solver=LP_RCPSP_Solver.CBC)
    solver.init_model()
    results_storage: ResultStorage = solver.solve(
        parameters_milp=ParametersMilp.default()
    )
    solution, fit = results_storage.get_best_solution_fit()
    solution_rebuilt = RCPSPSolution(
        problem=rcpsp_problem, rcpsp_permutation=solution.rcpsp_permutation
    )
    fit_2 = rcpsp_problem.evaluate(solution_rebuilt)
    print(rcpsp_problem.evaluate(solution), fit_2)
    print("Satisfy : ", rcpsp_problem.satisfy(solution))
    plot_resource_individual_gantt(rcpsp_problem, solution)
    plot_ressource_view(rcpsp_problem, solution)
    plt.show()


def single_mode_rcpsp_lp_cbc_robot():
    file = [f for f in files_available if "j1201_1.sm" in f][0]
    rcpsp_problem: SingleModeRCPSPModel = parse_file(file)
    original_duration_multiplier = 1
    n_assisted_activities = 30
    n_assistants = 3
    probability_of_cross_helper_precedences = 0.5
    fixed_helper_duration = 2
    random_seed = 4
    rcpsp_h = generate_rcpsp_with_helper_tasks_data(
        rcpsp_problem,
        original_duration_multiplier,
        n_assisted_activities,
        n_assistants,
        probability_of_cross_helper_precedences,
        fixed_helper_duration=fixed_helper_duration,
        random_seed=random_seed,
    )
    graph = rcpsp_h.compute_graph()
    cycles = graph.check_loop()
    print("cycles : ", cycles)
    predecessors = graph.precedessors_nodes(rcpsp_h.n_jobs + 2)
    print(len(predecessors), "expected : ", rcpsp_h.n_jobs + 1)
    solver = LP_RCPSP(rcpsp_model=rcpsp_h, lp_solver=LP_RCPSP_Solver.GRB)
    solver.init_model()
    results_storage: ResultStorage = solver.solve(
        parameters_milp=ParametersMilp.default()
    )
    solution, fit = results_storage.get_best_solution_fit()
    print("fit : ", fit)
    print(results_storage.list_solution_fits)
    solution_rebuilt = RCPSPSolution(
        problem=rcpsp_h, rcpsp_permutation=solution.rcpsp_permutation
    )
    fit_2 = rcpsp_h.evaluate(solution_rebuilt)
    print(rcpsp_h.evaluate(solution), fit_2)
    print("Satisfy : ", rcpsp_h.satisfy(solution))
    plot_resource_individual_gantt(rcpsp_h, solution)
    plot_ressource_view(rcpsp_h, solution)
    plt.show()


def multi_mode_rcpsp_lp_grb():
    file = [f for f in files_available if "j1010_1.mm" in f][0]
    rcpsp_problem: MultiModeRCPSPModel = parse_file(file)
    rcpsp_problem.set_fixed_modes([1 for i in range(rcpsp_problem.n_jobs)])
    solver = LP_MRCPSP(rcpsp_model=rcpsp_problem, lp_solver=LP_RCPSP_Solver.GRB)
    solver.init_model(greedy_start=False)
    results_storage: ResultStorage = solver.solve(
        parameters_milp=ParametersMilp.default()
    )
    solution, fit = results_storage.get_best_solution_fit()
    solution_rebuilt = RCPSPSolution(
        problem=rcpsp_problem,
        rcpsp_modes=solution.rcpsp_modes,
        rcpsp_permutation=solution.rcpsp_permutation,
    )
    fit_2 = rcpsp_problem.evaluate(solution_rebuilt)
    print(rcpsp_problem.evaluate(solution), fit_2)
    print("Satisfy : ", rcpsp_problem.satisfy(solution))
    print("From LP ", solution)
    print("Rebuilt : ", solution_rebuilt)
    plot_resource_individual_gantt(rcpsp_problem, solution)
    plot_ressource_view(rcpsp_problem, solution)
    plt.show()


def single_mode_rcpsp_lp_grb():
    file = [f for f in files_available if "j301_1.sm" in f][0]
    rcpsp_problem: SingleModeRCPSPModel = parse_file(file)
    solver = LP_MRCPSP(rcpsp_model=rcpsp_problem, lp_solver=LP_RCPSP_Solver.GRB)
    solver.init_model()
    store = solver.solve(parameters_milp=ParametersMilp.default())
    solution = store.get_best_solution()
    print(solution)
    print(solution.rcpsp_permutation)
    solution_rebuilt = RCPSPSolution(
        problem=rcpsp_problem, rcpsp_permutation=solution.rcpsp_permutation
    )
    fit_2 = rcpsp_problem.evaluate(solution_rebuilt)
    print(fit_2)
    print("Satisfy : ", rcpsp_problem.satisfy(solution))
    print("Initial = ")
    print("Rebuilt = ", solution_rebuilt)
    plot_resource_individual_gantt(rcpsp_problem, solution)
    plot_ressource_view(rcpsp_problem, solution)
    plt.show()


def single_mode_rcpsp_lp_grb_partial():
    file = [f for f in files_available if "j601_2.sm" in f][0]
    rcpsp_problem: SingleModeRCPSPModel = parse_file(file)
    dummy_solution = rcpsp_problem.get_dummy_solution()
    some_constraints = {
        task: dummy_solution.rcpsp_schedule[task]["start_time"] for task in [1, 2, 3, 4]
    }
    partial_solution = PartialSolution(task_mode=None, start_times=some_constraints)
    partial_solution_for_lp = partial_solution
    solver = LP_MRCPSP(rcpsp_model=rcpsp_problem, lp_solver=LP_RCPSP_Solver.GRB)
    solver.init_model(partial_solution=partial_solution_for_lp)
    params_milp = ParametersMilp.default()
    params_milp.TimeLimit = 30
    store = solver.solve(parameters_milp=ParametersMilp.default())
    solution, fit = store.get_best_solution_fit()
    print(solution)
    print("Constraint given as partial solution : ", partial_solution.start_times)
    print(
        "Found solution : ",
        {j: solution.rcpsp_schedule[j]["start_time"] for j in some_constraints},
    )
    print("Satisfy : ", rcpsp_problem.satisfy(solution))
    plot_resource_individual_gantt(rcpsp_problem, solution)
    plot_ressource_view(rcpsp_problem, solution)
    plt.show()


if __name__ == "__main__":
    single_mode_rcpsp_lp_cbc()
    # multi_mode_rcpsp_lp_grb()
    # single_mode_rcpsp_lp_cbc_robot()
    # single_mode_rcpsp_lp_grb()
    # single_mode_rcpsp_lp_lns()
    # multi_mode_rcpsp_cp()
