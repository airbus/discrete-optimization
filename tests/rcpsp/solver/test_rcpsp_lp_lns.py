import matplotlib.pyplot as plt

from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveHandling,
    ParamsObjectiveFunction,
    get_default_objective_setup,
)
from discrete_optimization.generic_tools.lns_mip import LNS_MILP
from discrete_optimization.generic_tools.lp_tools import ParametersMilp
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.rcpsp_model import (
    MethodBaseRobustification,
    MethodRobustification,
    MultiModeRCPSPModel,
    RCPSPModel,
    RCPSPSolution,
    SingleModeRCPSPModel,
    UncertainRCPSPModel,
    create_poisson_laws_duration,
)
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file
from discrete_optimization.rcpsp.rcpsp_utils import (
    plot_resource_individual_gantt,
    plot_ressource_view,
)
from discrete_optimization.rcpsp.solver.rcpsp_lp_lns_solver import (
    ConstraintHandlerFixStartTime,
    ConstraintHandlerStartTimeInterval,
    ConstraintHandlerStartTimeIntervalMRCPSP,
    InitialMethodRCPSP,
    InitialSolutionRCPSP,
)
from discrete_optimization.rcpsp.solver.rcpsp_lp_solver import (
    LP_MRCPSP,
    LP_RCPSP,
    LP_RCPSP_Solver,
)


def test_lns_sm():
    files_available = get_data_available()
    file = [f for f in files_available if "j301_1.sm" in f][0]
    rcpsp_problem: SingleModeRCPSPModel = parse_file(file)
    solver = LP_RCPSP(rcpsp_model=rcpsp_problem, lp_solver=LP_RCPSP_Solver.CBC)
    solver.init_model(greedy_start=False)
    parameters_milp = ParametersMilp(
        time_limit=10,
        pool_solutions=1000,
        mip_gap_abs=0.001,
        mip_gap=0.001,
        retrieve_all_solution=True,
        n_solutions_max=100,
    )
    params_objective_function = get_default_objective_setup(problem=rcpsp_problem)
    # constraint_handler = ConstraintHandlerFixStartTime(problem=rcpsp_problem,
    #                                                    fraction_fix_start_time=0.5)
    constraint_handler = ConstraintHandlerStartTimeInterval(
        problem=rcpsp_problem, fraction_to_fix=0.8, minus_delta=5, plus_delta=5
    )
    initial_solution_provider = InitialSolutionRCPSP(
        problem=rcpsp_problem,
        initial_method=InitialMethodRCPSP.DUMMY,
        params_objective_function=params_objective_function,
    )
    lns_solver = LNS_MILP(
        problem=rcpsp_problem,
        milp_solver=solver,
        initial_solution_provider=initial_solution_provider,
        constraint_handler=constraint_handler,
        params_objective_function=params_objective_function,
    )
    result_store = lns_solver.solve_lns(
        parameters_milp=parameters_milp, nb_iteration_lns=10
    )
    solution, fit = result_store.get_best_solution_fit()
    solution_rebuilt = RCPSPSolution(
        problem=rcpsp_problem, rcpsp_permutation=solution.rcpsp_permutation
    )
    fit_2 = rcpsp_problem.evaluate(solution_rebuilt)
    assert rcpsp_problem.evaluate(solution) == fit_2
    assert rcpsp_problem.satisfy(solution)
    plot_resource_individual_gantt(rcpsp_problem, solution)
    plot_ressource_view(rcpsp_problem, solution)


def test_lns_mm():
    files_available = get_data_available()
    # file = [f for f in files_available if 'j1010_8.mm' in f][0]
    file = [f for f in files_available if "j1201_1.sm" in f][0]
    rcpsp_problem: RCPSPModel = parse_file(file)
    if isinstance(rcpsp_problem, MultiModeRCPSPModel):
        rcpsp_problem.set_fixed_modes([1 for i in range(rcpsp_problem.n_jobs)])
    params_objective_function = get_default_objective_setup(problem=rcpsp_problem)
    params_objective_function = ParamsObjectiveFunction(
        objectives=["makespan"],
        weights=[-1],
        objective_handling=ObjectiveHandling.AGGREGATE,
        sense_function=ModeOptim.MAXIMIZATION,
    )
    solver = LP_MRCPSP(
        rcpsp_model=rcpsp_problem,
        lp_solver=LP_RCPSP_Solver.CBC,
        params_objective_function=params_objective_function,
    )
    solver.init_model(greedy_start=False)
    parameters_milp = ParametersMilp(
        time_limit=10,
        pool_solutions=1000,
        mip_gap_abs=0.001,
        mip_gap=0.001,
        retrieve_all_solution=True,
        n_solutions_max=100,
    )
    constraint_handler = ConstraintHandlerFixStartTime(
        problem=rcpsp_problem, fraction_fix_start_time=0.3
    )
    constraint_handler = ConstraintHandlerStartTimeIntervalMRCPSP(
        problem=rcpsp_problem, fraction_to_fix=0.5, minus_delta=2, plus_delta=2
    )
    initial_solution_provider = InitialSolutionRCPSP(
        problem=rcpsp_problem,
        initial_method=InitialMethodRCPSP.DUMMY,
        params_objective_function=params_objective_function,
    )
    lns_solver = LNS_MILP(
        problem=rcpsp_problem,
        milp_solver=solver,
        initial_solution_provider=initial_solution_provider,
        constraint_handler=constraint_handler,
        params_objective_function=params_objective_function,
    )
    result_store = lns_solver.solve_lns(
        parameters_milp=parameters_milp,
        nb_iteration_lns=10,
        skip_first_iteration=False,
    )
    solution, fit = result_store.get_best_solution_fit()
    solution_rebuilt = RCPSPSolution(
        problem=rcpsp_problem,
        rcpsp_permutation=solution.rcpsp_permutation,
        rcpsp_modes=solution.rcpsp_modes,
    )
    fit_2 = rcpsp_problem.evaluate(solution_rebuilt)
    assert rcpsp_problem.evaluate(solution) == fit_2
    assert rcpsp_problem.satisfy(solution)
    plot_resource_individual_gantt(rcpsp_problem, solution)
    plot_ressource_view(rcpsp_problem, solution)


if __name__ == "__main__":
    test_lns_sm()
