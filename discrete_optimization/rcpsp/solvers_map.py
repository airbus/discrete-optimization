#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any, Union

from discrete_optimization.generic_rcpsp_tools.solvers import GenericRcpspSolver
from discrete_optimization.generic_rcpsp_tools.solvers.gphh import (
    GphhGenericRcpspSolver,
)
from discrete_optimization.generic_rcpsp_tools.solvers.lns_cp import (
    LnsCpMznGenericRcpspSolver,
)
from discrete_optimization.generic_rcpsp_tools.solvers.ls import (
    LsGenericRcpspSolver,
    LsSolverType,
)
from discrete_optimization.generic_rcpsp_tools.typing import ANY_CLASSICAL_RCPSP
from discrete_optimization.generic_tools.cp_tools import CpSolverName, ParametersCp
from discrete_optimization.generic_tools.ea.ga_tools import (
    ParametersAltGa,
    ParametersGa,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp.problem_preemptive import PreemptiveRcpspProblem
from discrete_optimization.rcpsp.problem_specialized_constraints import (
    SpecialConstraintsPreemptiveRcpspProblem,
)
from discrete_optimization.rcpsp.solvers import RcpspSolver
from discrete_optimization.rcpsp.solvers.cp_mzn import (
    CpMultimodePreemptiveRcpspSolver,
    CpMultimodeRcpspSolver,
    CpPreemptiveRcpspSolver,
    CpRcpspSolver,
)
from discrete_optimization.rcpsp.solvers.cpm import CpmRcpspSolver
from discrete_optimization.rcpsp.solvers.cpsat import CpSatRcpspSolver
from discrete_optimization.rcpsp.solvers.ga import GaMultimodeRcpspSolver, GaRcpspSolver
from discrete_optimization.rcpsp.solvers.lp import (
    MathOptMultimodeRcpspSolver,
    MathOptRcpspSolver,
)
from discrete_optimization.rcpsp.solvers.pile import (
    PileCalendarRcpspSolver,
    PileRcpspSolver,
)

solvers: dict[
    str, list[tuple[Union[type[RcpspSolver], type[GenericRcpspSolver]], dict[str, Any]]]
] = {
    "lp": [
        (
            MathOptRcpspSolver,
            {},
        ),
        (
            MathOptMultimodeRcpspSolver,
            {},
        ),
    ],
    "greedy": [
        (PileRcpspSolver, {}),
        (PileCalendarRcpspSolver, {}),
    ],
    "cp": [
        (CpSatRcpspSolver, {"parameters_cp": ParametersCp.default()}),
        (
            CpRcpspSolver,
            {},
        ),
        (
            CpMultimodeRcpspSolver,
            {},
        ),
        (
            CpPreemptiveRcpspSolver,
            {},
        ),
        (
            CpMultimodePreemptiveRcpspSolver,
            {},
        ),
    ],
    "critical-path": [(CpmRcpspSolver, {})],
    "lns-scheduling": [
        (
            LnsCpMznGenericRcpspSolver,
            {
                "nb_iteration_lns": 100,
                "nb_iteration_no_improvement": 100,
                "parameters_cp": ParametersCp.default_fast_lns(),
                "cp_solver_name": CpSolverName.CHUFFED,
            },
        )
    ],
    "ls": [
        (LsGenericRcpspSolver, {"ls_solver": LsSolverType.SA, "nb_iteration_max": 2000})
    ],
    "ga": [
        (GaRcpspSolver, {"parameters_ga": ParametersGa.default_rcpsp()}),
        (GaMultimodeRcpspSolver, {"parameters_ga": ParametersAltGa.default_mrcpsp()}),
    ],
    "gphh": [(GphhGenericRcpspSolver, {})],
}

solvers_map = {}
for key in solvers:
    for solver, param in solvers[key]:
        solvers_map[solver] = (key, param)

solvers_compatibility: dict[
    Union[type[RcpspSolver], type[GenericRcpspSolver]], list[type[ANY_CLASSICAL_RCPSP]]
] = {
    MathOptRcpspSolver: [RcpspProblem],
    MathOptMultimodeRcpspSolver: [
        RcpspProblem,
    ],
    PileRcpspSolver: [RcpspProblem],
    PileCalendarRcpspSolver: [
        RcpspProblem,
    ],
    CpRcpspSolver: [RcpspProblem],
    CpMultimodeRcpspSolver: [
        RcpspProblem,
    ],
    CpPreemptiveRcpspSolver: [PreemptiveRcpspProblem],
    CpMultimodePreemptiveRcpspSolver: [PreemptiveRcpspProblem],
    LsGenericRcpspSolver: [
        PreemptiveRcpspProblem,
        SpecialConstraintsPreemptiveRcpspProblem,
        RcpspProblem,
    ],
    GaRcpspSolver: [
        PreemptiveRcpspProblem,
        SpecialConstraintsPreemptiveRcpspProblem,
        RcpspProblem,
    ],
    GaMultimodeRcpspSolver: [
        PreemptiveRcpspProblem,
        SpecialConstraintsPreemptiveRcpspProblem,
        RcpspProblem,
    ],
    LnsCpMznGenericRcpspSolver: [
        PreemptiveRcpspProblem,
        SpecialConstraintsPreemptiveRcpspProblem,
        RcpspProblem,
    ],
    CpmRcpspSolver: [
        PreemptiveRcpspProblem,
        SpecialConstraintsPreemptiveRcpspProblem,
        RcpspProblem,
    ],
    GphhGenericRcpspSolver: [
        PreemptiveRcpspProblem,
        SpecialConstraintsPreemptiveRcpspProblem,
        RcpspProblem,
    ],
}


def look_for_solver(
    domain: ANY_CLASSICAL_RCPSP,
) -> list[Union[type[RcpspSolver], type[GenericRcpspSolver]]]:
    class_domain = domain.__class__
    return look_for_solver_class(class_domain)


def look_for_solver_class(
    class_domain: type[ANY_CLASSICAL_RCPSP],
) -> list[Union[type[RcpspSolver], type[GenericRcpspSolver]]]:
    available = []
    for solver in solvers_compatibility:
        if class_domain in solvers_compatibility[solver]:
            available += [solver]
    return available


def solve(
    method: Union[type[RcpspSolver], type[GenericRcpspSolver]],
    problem: ANY_CLASSICAL_RCPSP,
    **kwargs: Any,
) -> ResultStorage:
    solver = return_solver(method=method, problem=problem, **kwargs)
    return solver.solve(**kwargs)


def solve_return_solver(
    method: Union[type[RcpspSolver], type[GenericRcpspSolver]],
    problem: ANY_CLASSICAL_RCPSP,
    **kwargs: Any,
) -> tuple[ResultStorage, Union[RcpspSolver, GenericRcpspSolver]]:
    solver = return_solver(method=method, problem=problem, **kwargs)
    return solver.solve(**kwargs), solver


def return_solver(
    method: Union[type[RcpspSolver], type[GenericRcpspSolver]],
    problem: ANY_CLASSICAL_RCPSP,
    **kwargs: Any,
) -> Union[RcpspSolver, GenericRcpspSolver]:
    solver: Union[RcpspSolver, GenericRcpspSolver]
    solver = method(problem=problem, **kwargs)
    try:
        solver.init_model(**kwargs)
    except:
        pass
    return solver


def get_solver_default_arguments(
    method: Union[type[RcpspSolver], type[GenericRcpspSolver]]
) -> dict[str, Any]:
    try:
        return solvers_map[method][1]
    except KeyError:
        raise KeyError(f"{method} is not in the list of available solvers for RCPSP.")
