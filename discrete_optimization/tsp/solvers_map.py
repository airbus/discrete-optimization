#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any

from discrete_optimization.generic_tools.cp_tools import CpSolverName
from discrete_optimization.generic_tools.do_problem import Problem
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.tsp.problem import (
    DistanceMatrixTspProblem,
    Point2DTspProblem,
    TspProblem,
)
from discrete_optimization.tsp.solvers import TspSolver
from discrete_optimization.tsp.solvers.cp_mzn import CPTspModel, CpTspSolver
from discrete_optimization.tsp.solvers.cpsat import CpSatTspSolver
from discrete_optimization.tsp.solvers.lp_iterative import (
    LPIterativeTspSolver,
    MILPSolver,
)
from discrete_optimization.tsp.solvers.ortools_routing import ORtoolsTspSolver

solvers: dict[str, list[tuple[type[TspSolver], dict[str, Any]]]] = {
    "lp": [
        (
            LPIterativeTspSolver,
            {"method": MILPSolver.CBC, "nb_iteration_max": 20, "plot": False},
        )
    ],
    "ortools": [(ORtoolsTspSolver, {}), (CpSatTspSolver, {})],
    "cp": [
        (
            CpTspSolver,
            {
                "model_type": CPTspModel.INT_VERSION,
                "cp_solver_name": CpSolverName.CHUFFED,
            },
        )
    ],
}

solvers_map = {}
for key in solvers:
    for solver, param in solvers[key]:
        solvers_map[solver] = (key, param)

solvers_compatibility: dict[type[TspSolver], list[type[Problem]]] = {}
for x in solvers:
    for y in solvers[x]:
        solvers_compatibility[y[0]] = [Point2DTspProblem, DistanceMatrixTspProblem]


def look_for_solver(domain: TspProblem) -> list[type[TspSolver]]:
    class_domain = domain.__class__
    return look_for_solver_class(class_domain)


def look_for_solver_class(class_domain: type["TspProblem"]) -> list[type[TspSolver]]:
    available = []
    for solver in solvers_compatibility:
        if class_domain in solvers_compatibility[solver]:
            available += [solver]
    return available


def solve(method: type[TspSolver], problem: TspProblem, **kwargs: Any) -> ResultStorage:
    solver = method(problem=problem, **kwargs)
    try:
        solver.init_model(**kwargs)
    except AttributeError:
        pass
    return solver.solve(**kwargs)


def return_solver(
    method: type[TspSolver], problem: TspProblem, **kwargs: Any
) -> TspSolver:
    solver = method(problem, **kwargs)
    try:
        solver.init_model(**kwargs)
    except:
        pass
    return solver
