#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any

from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.vrp.solver.lp_vrp_iterative import VRPIterativeLP
from discrete_optimization.vrp.solver.solver_ortools import VrpORToolsSolver
from discrete_optimization.vrp.solver.vrp_solver import SolverVrp
from discrete_optimization.vrp.vrp_model import VrpProblem, VrpProblem2D

solvers: dict[str, list[tuple[type[SolverVrp], dict[str, Any]]]] = {
    "ortools": [(VrpORToolsSolver, {"time_limit": 100})],
    "lp": [
        (VRPIterativeLP, {}),
    ],
}

solvers_map = {}
for key in solvers:
    for solver, param in solvers[key]:
        solvers_map[solver] = (key, param)

solvers_compatibility: dict[type[SolverVrp], list[type[VrpProblem]]] = {}
for x in solvers:
    for y in solvers[x]:
        solvers_compatibility[y[0]] = [VrpProblem2D]


def look_for_solver(domain: VrpProblem) -> list[type[SolverVrp]]:
    class_domain = domain.__class__
    return look_for_solver_class(class_domain)


def look_for_solver_class(class_domain: type[VrpProblem]) -> list[type[SolverVrp]]:
    available = []
    for solver in solvers_compatibility:
        if class_domain in solvers_compatibility[solver]:
            available += [solver]
    return available


def solve(method: type[SolverVrp], problem: VrpProblem, **kwargs: Any) -> ResultStorage:
    solver = method(problem, **kwargs)
    try:
        solver.init_model(**kwargs)
    except:
        pass
    return solver.solve(**kwargs)


def return_solver(
    method: type[SolverVrp], problem: VrpProblem, **kwargs: Any
) -> SolverVrp:
    solver = method(problem, **kwargs)
    try:
        solver.init_model(**kwargs)
    except:
        pass
    return solver
