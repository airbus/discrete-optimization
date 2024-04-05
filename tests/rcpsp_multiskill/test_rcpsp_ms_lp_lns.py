#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest

from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveHandling,
    ParamsObjectiveFunction,
)
from discrete_optimization.generic_tools.lns_mip import LNS_MILP
from discrete_optimization.generic_tools.lp_tools import ParametersMilp
from discrete_optimization.rcpsp.rcpsp_solution import RCPSPSolution
from discrete_optimization.rcpsp.rcpsp_utils import (
    plot_resource_individual_gantt,
    plot_ressource_view,
)
from discrete_optimization.rcpsp.solver.rcpsp_pile import PileSolverRCPSP_Calendar
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.rcpsp_multiskill.solvers.lp_model import (
    LP_Solver_MRSCPSP,
    MilpSolverName,
)
from discrete_optimization.rcpsp_multiskill.solvers.ms_rcpsp_lp_lns_solver import (
    ConstraintHandlerStartTimeIntervalMRCPSP,
    InitialMethodRCPSP,
    InitialSolutionMS_RCPSP,
)


@pytest.mark.skip(
    reason="mip+gurobi not working with gurobi license coming from mere pip install"
)
def test_multiskill_imopse():
    params_objective_function = ParamsObjectiveFunction(
        objectives=["makespan"],
        weights=[-1],
        objective_handling=ObjectiveHandling.AGGREGATE,
        sense_function=ModeOptim.MAXIMIZATION,
    )
    file = [f for f in get_data_available() if "100_5_20_9_D3.def" in f][0]
    model, _ = parse_file(file)
    model_rcpsp = model.build_multimode_rcpsp_calendar_representative()
    graph = model_rcpsp.compute_graph()
    cycles = graph.check_loop()
    solver = PileSolverRCPSP_Calendar(problem=model_rcpsp)
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 200
    store_solution = solver.solve(parameters_cp=parameters_cp)
    best_mrcpsp, fit = store_solution.get_best_solution_fit()
    solver = LP_Solver_MRSCPSP(
        problem=model,
        lp_solver=MilpSolverName.GRB,
        # CBC is not working well at all. -> so in unit test you should probably skip this test.
        params_objective_function=params_objective_function,
    )
    solver.init_model(max_time=600)
    parameters_milp = ParametersMilp(
        time_limit=30,
        pool_solutions=1000,
        mip_gap_abs=0.001,
        mip_gap=0.001,
        retrieve_all_solution=True,
        n_solutions_max=100,
    )
    constraint_handler = ConstraintHandlerStartTimeIntervalMRCPSP(
        problem=model, fraction_to_fix=0.95, minus_delta=5, plus_delta=5
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
        nb_iteration_lns=10,
        nb_iteration_no_improvement=10,
        callbacks=[TimerStopper(total_seconds=200)],
        skip_first_iteration=False,
    )
    solution, fit = result_store.get_best_solution_fit()
    model.evaluate(solution)
    assert model.satisfy(solution)
