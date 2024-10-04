#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest

from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveHandling,
    ParamsObjectiveFunction,
    get_default_objective_setup,
)
from discrete_optimization.generic_tools.lns_mip import LNS_MILP
from discrete_optimization.generic_tools.lp_tools import ParametersMilp
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file
from discrete_optimization.rcpsp.rcpsp_solution import RCPSPSolution
from discrete_optimization.rcpsp.rcpsp_utils import (
    plot_resource_individual_gantt,
    plot_ressource_view,
)
from discrete_optimization.rcpsp.solver.rcpsp_lp_lns_solver import (
    ConstraintHandlerFixStartTime_MathOpt,
    ConstraintHandlerStartTimeInterval_MathOpt,
    ConstraintHandlerStartTimeIntervalMRCPSP_GRB,
    ConstraintHandlerStartTimeIntervalMRCPSP_MathOpt,
    InitialMethodRCPSP,
    InitialSolutionRCPSP,
)
from discrete_optimization.rcpsp.solver.rcpsp_lp_solver import (
    LP_MRCPSP_GUROBI,
    LP_MRCPSP_MATHOPT,
    LP_RCPSP_MATHOPT,
)


@pytest.mark.parametrize(
    "constraint_handler_cls",
    [ConstraintHandlerStartTimeInterval_MathOpt, ConstraintHandlerFixStartTime_MathOpt],
)
def test_lns_sm(constraint_handler_cls):
    files_available = get_data_available()
    file = [f for f in files_available if "j301_1.sm" in f][0]
    rcpsp_problem = parse_file(file)
    solver = LP_RCPSP_MATHOPT(problem=rcpsp_problem)
    solver.init_model(greedy_start=False)
    parameters_milp = ParametersMilp(
        pool_solutions=1000,
        mip_gap_abs=0.001,
        mip_gap=0.001,
        retrieve_all_solution=True,
    )
    params_objective_function = get_default_objective_setup(problem=rcpsp_problem)
    constraint_handler = constraint_handler_cls(problem=rcpsp_problem)
    initial_solution_provider = InitialSolutionRCPSP(
        problem=rcpsp_problem,
        initial_method=InitialMethodRCPSP.DUMMY,
        params_objective_function=params_objective_function,
    )
    lns_solver = LNS_MILP(
        problem=rcpsp_problem,
        subsolver=solver,
        initial_solution_provider=initial_solution_provider,
        constraint_handler=constraint_handler,
        params_objective_function=params_objective_function,
    )
    result_store = lns_solver.solve(
        time_limit_subsolver=10, parameters_milp=parameters_milp, nb_iteration_lns=10
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


@pytest.mark.parametrize(
    "solver_cls, constraint_handler_cls",
    [
        (LP_MRCPSP_MATHOPT, ConstraintHandlerStartTimeIntervalMRCPSP_MathOpt),
        # (LP_MRCPSP_GUROBI, ConstraintHandlerStartTimeIntervalMRCPSP_GRB),  # skip as model too large for free license
    ],
)
def test_lns_mm(solver_cls, constraint_handler_cls):
    files_available = get_data_available()
    file = [f for f in files_available if "j1010_1.mm" in f][0]
    rcpsp_problem: RCPSPModel = parse_file(file)
    if rcpsp_problem.is_rcpsp_multimode():
        rcpsp_problem.set_fixed_modes([1 for i in range(rcpsp_problem.n_jobs)])
    params_objective_function = ParamsObjectiveFunction(
        objectives=["makespan"],
        weights=[-1],
        objective_handling=ObjectiveHandling.AGGREGATE,
        sense_function=ModeOptim.MAXIMIZATION,
    )
    solver = solver_cls(
        problem=rcpsp_problem,
        params_objective_function=params_objective_function,
    )
    solver.init_model(greedy_start=False)
    parameters_milp = ParametersMilp(
        pool_solutions=1000,
        mip_gap_abs=0.001,
        mip_gap=0.001,
        retrieve_all_solution=True,
    )
    constraint_handler = constraint_handler_cls(
        problem=rcpsp_problem, fraction_to_fix=0.5, minus_delta=2, plus_delta=2
    )
    initial_solution_provider = InitialSolutionRCPSP(
        problem=rcpsp_problem,
        initial_method=InitialMethodRCPSP.DUMMY,
        params_objective_function=params_objective_function,
    )
    lns_solver = LNS_MILP(
        problem=rcpsp_problem,
        subsolver=solver,
        initial_solution_provider=initial_solution_provider,
        constraint_handler=constraint_handler,
        params_objective_function=params_objective_function,
    )
    result_store = lns_solver.solve(
        parameters_milp=parameters_milp,
        time_limit_subsolver=10,
        nb_iteration_lns=10,
        skip_initial_solution_provider=False,
    )
    solution, fit = result_store.get_best_solution_fit()
    assert rcpsp_problem.satisfy(solution)
    plot_resource_individual_gantt(rcpsp_problem, solution)
    plot_ressource_view(rcpsp_problem, solution)


if __name__ == "__main__":
    test_lns_sm()
