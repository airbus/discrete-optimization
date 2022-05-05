import time

import matplotlib.pyplot as plt
from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveHandling,
    ParamsObjectiveFunction,
)
from discrete_optimization.generic_tools.lns_mip import LNS_MILP
from discrete_optimization.generic_tools.lp_tools import ParametersMilp
from discrete_optimization.rcpsp.rcpsp_model import RCPSPSolution
from discrete_optimization.rcpsp.rcpsp_utils import (
    plot_resource_individual_gantt,
    plot_ressource_view,
)
from discrete_optimization.rcpsp.solver import CP_MRCPSP_MZN
from discrete_optimization.rcpsp_multiskill.solvers.lp_model import (
    LP_Solver_MRSCPSP,
    MilpSolverName,
)
from discrete_optimization.rcpsp_multiskill.solvers.ms_rcpsp_lp_lns_solver import (
    ConstraintHandlerStartTimeIntervalMRCPSP,
    InitialMethodRCPSP,
    InitialSolutionMS_RCPSP,
)

from tests.rcpsp_multiskills.solvers.instance_creator import create_ms_rcpsp_demo


def lns_multikill():
    params_objective_function = ParamsObjectiveFunction(
        objectives=["makespan"],
        weights=[-1],
        objective_handling=ObjectiveHandling.AGGREGATE,
        sense_function=ModeOptim.MAXIMIZATION,
    )
    model, model_rcpsp = create_ms_rcpsp_demo()
    solver = CP_MRCPSP_MZN(rcpsp_model=model_rcpsp)
    store_solution = solver.solve(parameters_cp=ParametersCP.default())
    best_mrcpsp, fit = store_solution.get_best_solution_fit()
    print("Optim found by mrcpsp solver : ", fit)
    solver = LP_Solver_MRSCPSP(
        rcpsp_model=model,
        lp_solver=MilpSolverName.GRB,
        params_objective_function=params_objective_function,
    )
    t = time.time()
    print("init LP model...")
    solver.init_model()
    print("model LP initialized in ", time.time() - t, " seconds")
    parameters_milp = ParametersMilp(
        time_limit=200,
        pool_solutions=1000,
        mip_gap_abs=0.001,
        mip_gap=0.001,
        retrieve_all_solution=True,
        n_solutions_max=100,
    )
    constraint_handler = ConstraintHandlerStartTimeIntervalMRCPSP(
        problem=model, fraction_to_fix=0.5, minus_delta=5, plus_delta=5
    )
    initial_solution_provider = InitialSolutionMS_RCPSP(
        problem=model,
        initial_method=InitialMethodRCPSP.PILE_CALENDAR,
        params_objective_function=params_objective_function,
    )
    lns_solver = LNS_MILP(
        problem=model,
        milp_solver=solver,
        initial_solution_provider=initial_solution_provider,
        constraint_handler=constraint_handler,
        params_objective_function=params_objective_function,
    )
    result_store = lns_solver.solve_lns(
        parameters_milp=parameters_milp,
        nb_iteration_lns=300,
        nb_iteration_no_improvement=200,
        max_time_seconds=60 * 60,
        skip_first_iteration=False,
    )
    solution, fit = result_store.get_best_solution_fit()
    print("Fitness : ", model.evaluate(solution))
    print("Satisfiable  : ", model.satisfy(solution))
    rebuilt_sol_rcpsp = RCPSPSolution(
        problem=model_rcpsp,
        rcpsp_permutation=None,
        rcpsp_schedule=solution.schedule,
        rcpsp_modes=[solution.modes[x] for x in range(2, model.n_jobs_non_dummy + 2)],
    )
    plot_ressource_view(model_rcpsp, rebuilt_sol_rcpsp)
    plot_resource_individual_gantt(
        rcpsp_model=model_rcpsp,
        rcpsp_sol=rebuilt_sol_rcpsp,
        resource_types_to_consider=["worker"],
    )
    plt.show()


def multiskill_imopse():
    from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill_parser import (
        get_data_available,
        parse_file,
    )

    params_objective_function = ParamsObjectiveFunction(
        objectives=["makespan"],
        weights=[-1],
        objective_handling=ObjectiveHandling.AGGREGATE,
        sense_function=ModeOptim.MAXIMIZATION,
    )
    file = [f for f in get_data_available() if "100_5_20_9_D3.def" in f][0]
    model = parse_file(file)
    model_rcpsp = model.build_multimode_rcpsp_calendar_representative()
    graph = model_rcpsp.compute_graph()
    cycles = graph.check_loop()
    print(cycles)
    from discrete_optimization.rcpsp.solver.rcpsp_pile import PileSolverRCPSP_Calendar

    # solver = CP_MRCPSP_MZN(rcpsp_model=model_rcpsp)
    solver = PileSolverRCPSP_Calendar(rcpsp_model=model_rcpsp)
    store_solution = solver.solve(parameters_cp=ParametersCP.default())
    best_mrcpsp, fit = store_solution.get_best_solution_fit()
    print("Optim found by mrcpsp solver : ", fit)
    solver = LP_Solver_MRSCPSP(
        rcpsp_model=model,
        lp_solver=MilpSolverName.GRB,
        params_objective_function=params_objective_function,
    )
    t = time.time()
    print("init LP model...")
    solver.init_model()
    print("model LP initialized in ", time.time() - t, " seconds")
    parameters_milp = ParametersMilp(
        time_limit=200,
        pool_solutions=1000,
        mip_gap_abs=0.001,
        mip_gap=0.001,
        retrieve_all_solution=True,
        n_solutions_max=100,
    )
    constraint_handler = ConstraintHandlerStartTimeIntervalMRCPSP(
        problem=model, fraction_to_fix=0.5, minus_delta=5, plus_delta=5
    )
    initial_solution_provider = InitialSolutionMS_RCPSP(
        problem=model,
        initial_method=InitialMethodRCPSP.PILE_CALENDAR,
        params_objective_function=params_objective_function,
    )
    lns_solver = LNS_MILP(
        problem=model,
        milp_solver=solver,
        initial_solution_provider=initial_solution_provider,
        constraint_handler=constraint_handler,
        params_objective_function=params_objective_function,
    )
    result_store = lns_solver.solve_lns(
        parameters_milp=parameters_milp,
        nb_iteration_lns=300,
        nb_iteration_no_improvement=200,
        max_time_seconds=60 * 60,
        skip_first_iteration=False,
    )
    solution, fit = result_store.get_best_solution_fit()
    print("Fitness : ", model.evaluate(solution))
    print("Satisfiable  : ", model.satisfy(solution))
    rebuilt_sol_rcpsp = RCPSPSolution(
        problem=model_rcpsp,
        rcpsp_permutation=None,
        rcpsp_schedule=solution.schedule,
        rcpsp_modes=[solution.modes[x] for x in range(2, model.n_jobs_non_dummy + 2)],
    )
    plot_ressource_view(model_rcpsp, rebuilt_sol_rcpsp)
    plot_resource_individual_gantt(
        rcpsp_model=model_rcpsp,
        rcpsp_sol=rebuilt_sol_rcpsp,
        resource_types_to_consider=["worker"],
    )
    plt.show()


if __name__ == "__main__":
    multiskill_imopse()
