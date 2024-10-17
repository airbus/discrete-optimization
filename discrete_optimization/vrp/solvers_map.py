#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any

from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.vrp.problem import Customer2DVrpProblem, VrpProblem
from discrete_optimization.vrp.solvers import VrpSolver
from discrete_optimization.vrp.solvers.lp_iterative import LPIterativeVrpSolver
from discrete_optimization.vrp.solvers.ortools_routing import OrtoolsVrpSolver

solvers: dict[str, list[tuple[type[VrpSolver], dict[str, Any]]]] = {
    "ortools": [(OrtoolsVrpSolver, {"time_limit": 100})],
    "lp": [
        (LPIterativeVrpSolver, {}),
    ],
}

solvers_map = {}
for key in solvers:
    for solver, param in solvers[key]:
        solvers_map[solver] = (key, param)

solvers_compatibility: dict[type[VrpSolver], list[type[VrpProblem]]] = {}
for x in solvers:
    for y in solvers[x]:
        solvers_compatibility[y[0]] = [Customer2DVrpProblem]


def look_for_solver(domain: VrpProblem) -> list[type[VrpSolver]]:
    class_domain = domain.__class__
    return look_for_solver_class(class_domain)


def look_for_solver_class(class_domain: type[VrpProblem]) -> list[type[VrpSolver]]:
    available = []
    for solver in solvers_compatibility:
        if class_domain in solvers_compatibility[solver]:
            available += [solver]
    return available


def solve(method: type[VrpSolver], problem: VrpProblem, **kwargs: Any) -> ResultStorage:
    solver = method(problem, **kwargs)
    try:
        solver.init_model(**kwargs)
    except:
        pass
    return solver.solve(**kwargs)


def return_solver(
    method: type[VrpSolver], problem: VrpProblem, **kwargs: Any
) -> VrpSolver:
    solver = method(problem, **kwargs)
    try:
        solver.init_model(**kwargs)
    except:
        pass
    return solver
