#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_rcpsp_tools.large_neighborhood_search_scheduling import (
    LargeNeighborhoodSearchScheduling,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.generic_tools.ea.ga_tools import ParametersAltGa
from discrete_optimization.generic_tools.lp_tools import ParametersMilp
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.solver.cp_solvers import CPSolverName
from discrete_optimization.rcpsp.solver.gphh_solver import GPHH
from discrete_optimization.rcpsp.solver.ls_solver import LS_SOLVER, LS_RCPSP_Solver
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import (
    MS_RCPSPModel,
    MS_RCPSPModel_Variant,
)
from discrete_optimization.rcpsp_multiskill.solvers.calendar_solver_iterative import (
    SolverWithCalendarIterative,
)
from discrete_optimization.rcpsp_multiskill.solvers.cp_lns_solver import (
    LargeNeighborhoodSearchMSRCPSP,
)
from discrete_optimization.rcpsp_multiskill.solvers.cp_solvers import (
    CP_MS_MRCPSP_MZN,
    CP_MS_MRCPSP_MZN_PREEMPTIVE,
)
from discrete_optimization.rcpsp_multiskill.solvers.lp_model import (
    LP_Solver_MRSCPSP,
    MilpSolverName,
)
from discrete_optimization.rcpsp_multiskill.solvers.ms_rcpsp_cp_lns_solver import (
    LNS_CP_MS_RCPSP_SOLVER,
    OptionNeighbor,
)
from discrete_optimization.rcpsp_multiskill.solvers.ms_rcpsp_ga_solver import (
    GA_MSRCPSP_Solver,
)

solvers = {
    "lp": [
        (
            LP_Solver_MRSCPSP,
            {
                "lp_solver": MilpSolverName.CBC,
                "parameters_milp": ParametersMilp.default(),
            },
        )
    ],
    "cp": [
        (
            CP_MS_MRCPSP_MZN,
            {
                "cp_solver_name": CPSolverName.CHUFFED,
                "parameters_cp": ParametersCP.default(),
                "exact_skills_need": False,
            },
        ),
        (
            CP_MS_MRCPSP_MZN_PREEMPTIVE,
            {
                "cp_solver_name": CPSolverName.CHUFFED,
                "parameters_cp": ParametersCP.default(),
                "exact_skills_need": False,
                "nb_preemptive": 5,
            },
        ),
    ],
    "lns": [
        (
            LNS_CP_MS_RCPSP_SOLVER,
            {
                "nb_iteration_lns": 500,
                "option_neighbor": OptionNeighbor.MIX_LARGE_NEIGH,
                "exact_skills_need": False,
            },
        )
    ],
    "lns-cp": [
        (
            LNS_CP_MS_RCPSP_SOLVER,
            {
                "nb_iteration_lns": 20,
                "option_neighbor": OptionNeighbor.MIX_LARGE_NEIGH,
                "exact_skills_need": False,
            },
        )
    ],
    "ls": [(LS_RCPSP_Solver, {"ls_solver": LS_SOLVER.SA, "nb_iteration_max": 20})],
    "ga": [(GA_MSRCPSP_Solver, {"parameters_ga": ParametersAltGa.default_msrcpsp()})],
    "lns-cp-calendar": [
        (
            SolverWithCalendarIterative,
            {
                "option_neighbor": OptionNeighbor.MIX_LARGE_NEIGH,
                "parameters_cp": ParametersCP.default(),
                "nb_iteration_lns": 20,
                "skip_first_iteration": False,
            },
        )
    ],
    "lns-cp-rcpsp": [
        (
            LargeNeighborhoodSearchMSRCPSP,
            {
                "nb_iteration_lns": 100,
                "nb_iteration_no_improvement": 100,
                "parameters_cp": ParametersCP.default_free(),
                "cut_part": 6,
                "add_objective_makespan": False,
                "fraction_subproblem": 0.4,
            },
        )
    ],
    "lns-scheduling": [
        (
            LargeNeighborhoodSearchScheduling,
            {
                "nb_iteration_lns": 100,
                "nb_iteration_no_improvement": 100,
                "parameters_cp": ParametersCP.default(),
            },
        )
    ],
    "gphh": [(GPHH, {})],
}

solvers_map = {}
for key in solvers:
    for solver, param in solvers[key]:
        solvers_map[solver] = (key, param)

solvers_compatibility = {
    LP_Solver_MRSCPSP: [MS_RCPSPModel, MS_RCPSPModel_Variant],
    SolverWithCalendarIterative: [MS_RCPSPModel, MS_RCPSPModel_Variant],
    CP_MS_MRCPSP_MZN: [MS_RCPSPModel, MS_RCPSPModel_Variant],
    LNS_CP_MS_RCPSP_SOLVER: [MS_RCPSPModel, MS_RCPSPModel_Variant],
    LS_RCPSP_Solver: [MS_RCPSPModel, MS_RCPSPModel_Variant],
    GA_MSRCPSP_Solver: [MS_RCPSPModel_Variant],
    CP_MS_MRCPSP_MZN_PREEMPTIVE: [MS_RCPSPModel, MS_RCPSPModel_Variant],
    LargeNeighborhoodSearchMSRCPSP: [MS_RCPSPModel, MS_RCPSPModel_Variant],
    LargeNeighborhoodSearchScheduling: [MS_RCPSPModel, MS_RCPSPModel_Variant],
    GPHH: [MS_RCPSPModel, MS_RCPSPModel_Variant],
}


def look_for_solver(domain):
    class_domain = domain.__class__
    return look_for_solver_class(class_domain)


def look_for_solver_class(class_domain):
    available = []
    for solver in solvers_compatibility:
        if class_domain in solvers_compatibility[solver]:
            available += [solver]
    return available


def solve(method, rcpsp_model: MS_RCPSPModel, **args) -> ResultStorage:
    if method == GPHH:
        solver = GPHH([rcpsp_model], rcpsp_model, **args)
    else:
        solver = method(rcpsp_model, **args)
    try:
        solver.init_model(**args)
    except:
        pass
    return solver.solve(**args)


def return_solver(method, rcpsp_model: MS_RCPSPModel, **args) -> ResultStorage:
    if method == GPHH:
        solver = GPHH([rcpsp_model], rcpsp_model, **args)
    else:
        solver = method(rcpsp_model, **args)
    try:
        solver.init_model(**args)
    except:
        pass
    return solver
