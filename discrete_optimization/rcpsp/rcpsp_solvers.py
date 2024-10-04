#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any, Union

from discrete_optimization.generic_rcpsp_tools.generic_rcpsp_solver import (
    SolverGenericRCPSP,
)
from discrete_optimization.generic_rcpsp_tools.gphh_solver import GPHH
from discrete_optimization.generic_rcpsp_tools.large_neighborhood_search_scheduling import (
    LargeNeighborhoodSearchScheduling,
)
from discrete_optimization.generic_rcpsp_tools.ls_solver import (
    LS_SOLVER,
    LS_RCPSP_Solver,
)
from discrete_optimization.generic_rcpsp_tools.typing import ANY_CLASSICAL_RCPSP
from discrete_optimization.generic_tools.cp_tools import CPSolverName, ParametersCP
from discrete_optimization.generic_tools.ea.ga_tools import (
    ParametersAltGa,
    ParametersGa,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel
from discrete_optimization.rcpsp.rcpsp_model_preemptive import RCPSPModelPreemptive
from discrete_optimization.rcpsp.solver.cp_solvers import (
    CP_MRCPSP_MZN,
    CP_MRCPSP_MZN_PREEMPTIVE,
    CP_RCPSP_MZN,
    CP_RCPSP_MZN_PREEMPTIVE,
)
from discrete_optimization.rcpsp.solver.cpm import CPM
from discrete_optimization.rcpsp.solver.cpsat_solver import CPSatRCPSPSolver
from discrete_optimization.rcpsp.solver.rcpsp_ga_solver import (
    GA_MRCPSP_Solver,
    GA_RCPSP_Solver,
)
from discrete_optimization.rcpsp.solver.rcpsp_lp_solver import (
    LP_MRCPSP_MATHOPT,
    LP_RCPSP_MATHOPT,
)
from discrete_optimization.rcpsp.solver.rcpsp_pile import (
    PileSolverRCPSP,
    PileSolverRCPSP_Calendar,
)
from discrete_optimization.rcpsp.solver.rcpsp_solver import SolverRCPSP
from discrete_optimization.rcpsp.specialized_rcpsp.rcpsp_specialized_constraints import (
    RCPSPModelSpecialConstraintsPreemptive,
)

solvers: dict[
    str, list[tuple[Union[type[SolverRCPSP], type[SolverGenericRCPSP]], dict[str, Any]]]
] = {
    "lp": [
        (
            LP_RCPSP_MATHOPT,
            {},
        ),
        (
            LP_MRCPSP_MATHOPT,
            {},
        ),
    ],
    "greedy": [
        (PileSolverRCPSP, {}),
        (PileSolverRCPSP_Calendar, {}),
    ],
    "cp": [
        (CPSatRCPSPSolver, {"parameters_cp": ParametersCP.default()}),
        (
            CP_RCPSP_MZN,
            {},
        ),
        (
            CP_MRCPSP_MZN,
            {},
        ),
        (
            CP_RCPSP_MZN_PREEMPTIVE,
            {},
        ),
        (
            CP_MRCPSP_MZN_PREEMPTIVE,
            {},
        ),
    ],
    "critical-path": [(CPM, {})],
    "lns-scheduling": [
        (
            LargeNeighborhoodSearchScheduling,
            {
                "nb_iteration_lns": 100,
                "nb_iteration_no_improvement": 100,
                "parameters_cp": ParametersCP.default_fast_lns(),
                "cp_solver_name": CPSolverName.CHUFFED,
            },
        )
    ],
    "ls": [(LS_RCPSP_Solver, {"ls_solver": LS_SOLVER.SA, "nb_iteration_max": 2000})],
    "ga": [
        (GA_RCPSP_Solver, {"parameters_ga": ParametersGa.default_rcpsp()}),
        (GA_MRCPSP_Solver, {"parameters_ga": ParametersAltGa.default_mrcpsp()}),
    ],
    "gphh": [(GPHH, {})],
}

solvers_map = {}
for key in solvers:
    for solver, param in solvers[key]:
        solvers_map[solver] = (key, param)

solvers_compatibility: dict[
    Union[type[SolverRCPSP], type[SolverGenericRCPSP]], list[type[ANY_CLASSICAL_RCPSP]]
] = {
    LP_RCPSP_MATHOPT: [RCPSPModel],
    LP_MRCPSP_MATHOPT: [
        RCPSPModel,
    ],
    PileSolverRCPSP: [RCPSPModel],
    PileSolverRCPSP_Calendar: [
        RCPSPModel,
    ],
    CP_RCPSP_MZN: [RCPSPModel],
    CP_MRCPSP_MZN: [
        RCPSPModel,
    ],
    CP_RCPSP_MZN_PREEMPTIVE: [RCPSPModelPreemptive],
    CP_MRCPSP_MZN_PREEMPTIVE: [RCPSPModelPreemptive],
    LS_RCPSP_Solver: [
        RCPSPModelPreemptive,
        RCPSPModelSpecialConstraintsPreemptive,
        RCPSPModel,
    ],
    GA_RCPSP_Solver: [
        RCPSPModelPreemptive,
        RCPSPModelSpecialConstraintsPreemptive,
        RCPSPModel,
    ],
    GA_MRCPSP_Solver: [
        RCPSPModelPreemptive,
        RCPSPModelSpecialConstraintsPreemptive,
        RCPSPModel,
    ],
    LargeNeighborhoodSearchScheduling: [
        RCPSPModelPreemptive,
        RCPSPModelSpecialConstraintsPreemptive,
        RCPSPModel,
    ],
    CPM: [
        RCPSPModelPreemptive,
        RCPSPModelSpecialConstraintsPreemptive,
        RCPSPModel,
    ],
    GPHH: [
        RCPSPModelPreemptive,
        RCPSPModelSpecialConstraintsPreemptive,
        RCPSPModel,
    ],
}


def look_for_solver(
    domain: ANY_CLASSICAL_RCPSP,
) -> list[Union[type[SolverRCPSP], type[SolverGenericRCPSP]]]:
    class_domain = domain.__class__
    return look_for_solver_class(class_domain)


def look_for_solver_class(
    class_domain: type[ANY_CLASSICAL_RCPSP],
) -> list[Union[type[SolverRCPSP], type[SolverGenericRCPSP]]]:
    available = []
    for solver in solvers_compatibility:
        if class_domain in solvers_compatibility[solver]:
            available += [solver]
    return available


def solve(
    method: Union[type[SolverRCPSP], type[SolverGenericRCPSP]],
    problem: ANY_CLASSICAL_RCPSP,
    **kwargs: Any,
) -> ResultStorage:
    solver = return_solver(method=method, problem=problem, **kwargs)
    return solver.solve(**kwargs)


def solve_return_solver(
    method: Union[type[SolverRCPSP], type[SolverGenericRCPSP]],
    problem: ANY_CLASSICAL_RCPSP,
    **kwargs: Any,
) -> tuple[ResultStorage, Union[SolverRCPSP, SolverGenericRCPSP]]:
    solver = return_solver(method=method, problem=problem, **kwargs)
    return solver.solve(**kwargs), solver


def return_solver(
    method: Union[type[SolverRCPSP], type[SolverGenericRCPSP]],
    problem: ANY_CLASSICAL_RCPSP,
    **kwargs: Any,
) -> Union[SolverRCPSP, SolverGenericRCPSP]:
    solver: Union[SolverRCPSP, SolverGenericRCPSP]
    solver = method(problem=problem, **kwargs)
    try:
        solver.init_model(**kwargs)
    except:
        pass
    return solver


def get_solver_default_arguments(
    method: Union[type[SolverRCPSP], type[SolverGenericRCPSP]]
) -> dict[str, Any]:
    try:
        return solvers_map[method][1]
    except KeyError:
        raise KeyError(f"{method} is not in the list of available solvers for RCPSP.")
