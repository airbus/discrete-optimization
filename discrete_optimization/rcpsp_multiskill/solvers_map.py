#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_rcpsp_tools.solvers.gphh import (
    GphhGenericRcpspSolver,
)
from discrete_optimization.generic_rcpsp_tools.solvers.lns_cp import (
    LnsCpMznGenericRcpspSolver,
)
from discrete_optimization.generic_rcpsp_tools.solvers.ls import LsGenericRcpspSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp_multiskill.problem import (
    MultiskillRcpspProblem,
    VariantMultiskillRcpspProblem,
)
from discrete_optimization.rcpsp_multiskill.solvers.cp_mzn import (
    CpMultiskillRcpspSolver,
    CpPreemptiveMultiskillRcpspSolver,
)
from discrete_optimization.rcpsp_multiskill.solvers.cpsat import (
    CpSatMultiskillRcpspSolver,
)
from discrete_optimization.rcpsp_multiskill.solvers.ga import GaMultiskillRcpspSolver
from discrete_optimization.rcpsp_multiskill.solvers.lp import (
    MathOptMultiskillRcpspSolver,
)

solvers = {
    "lp": [
        (
            MathOptMultiskillRcpspSolver,
            {},
        )
    ],
    "cp": [
        (
            CpSatMultiskillRcpspSolver,
            {},
        ),
        (
            CpMultiskillRcpspSolver,
            {},
        ),
        (
            CpPreemptiveMultiskillRcpspSolver,
            {
                "nb_preemptive": 5,
            },
        ),
    ],
    "ls": [(LsGenericRcpspSolver, {"nb_iteration_max": 20})],
    "ga": [(GaMultiskillRcpspSolver, {})],
    "lns-scheduling": [
        (
            LnsCpMznGenericRcpspSolver,
            {
                "nb_iteration_lns": 100,
                "nb_iteration_no_improvement": 100,
            },
        )
    ],
    "gphh": [(GphhGenericRcpspSolver, {})],
}

solvers_map = {}
for key in solvers:
    for solver, param in solvers[key]:
        solvers_map[solver] = (key, param)

solvers_compatibility = {
    MathOptMultiskillRcpspSolver: [
        MultiskillRcpspProblem,
        VariantMultiskillRcpspProblem,
    ],
    CpMultiskillRcpspSolver: [MultiskillRcpspProblem, VariantMultiskillRcpspProblem],
    LsGenericRcpspSolver: [MultiskillRcpspProblem, VariantMultiskillRcpspProblem],
    GaMultiskillRcpspSolver: [VariantMultiskillRcpspProblem],
    CpPreemptiveMultiskillRcpspSolver: [
        MultiskillRcpspProblem,
        VariantMultiskillRcpspProblem,
    ],
    LnsCpMznGenericRcpspSolver: [MultiskillRcpspProblem, VariantMultiskillRcpspProblem],
    GphhGenericRcpspSolver: [MultiskillRcpspProblem, VariantMultiskillRcpspProblem],
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


def solve(method, problem: MultiskillRcpspProblem, **args) -> ResultStorage:
    solver = method(problem, **args)
    try:
        solver.init_model(**args)
    except:
        pass
    return solver.solve(**args)


def return_solver(method, problem: MultiskillRcpspProblem, **args) -> ResultStorage:
    solver = method(problem, **args)
    try:
        solver.init_model(**args)
    except:
        pass
    return solver
