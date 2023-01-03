#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Tuple, Type

from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.vrp.solver.lp_vrp_iterative import VRPIterativeLP
from discrete_optimization.vrp.solver.lp_vrp_iterative_pymip import VRPIterativeLP_Pymip
from discrete_optimization.vrp.solver.solver_ortools import VrpORToolsSolver
from discrete_optimization.vrp.solver.vrp_solver import SolverVrp
from discrete_optimization.vrp.vrp_model import VrpProblem, VrpProblem2D

solvers: Dict[str, List[Tuple[Type[SolverVrp], Dict[str, Any]]]] = {
    "ortools": [(VrpORToolsSolver, {"limit_time_s": 100})],
    "lp": [(VRPIterativeLP, {}), (VRPIterativeLP_Pymip, {})],
}

solvers_map = {}
for key in solvers:
    for solver, param in solvers[key]:
        solvers_map[solver] = (key, param)

solvers_compatibility: Dict[Type[SolverVrp], List[Type[VrpProblem]]] = {}
for x in solvers:
    for y in solvers[x]:
        solvers_compatibility[y[0]] = [VrpProblem2D]


def look_for_solver(domain: VrpProblem) -> List[Type[SolverVrp]]:
    class_domain = domain.__class__
    return look_for_solver_class(class_domain)


def look_for_solver_class(class_domain: Type[VrpProblem]) -> List[Type[SolverVrp]]:
    available = []
    for solver in solvers_compatibility:
        if class_domain in solvers_compatibility[solver]:
            available += [solver]
    return available


def solve(
    method: Type[SolverVrp], vrp_problem: VrpProblem, **kwargs: Any
) -> ResultStorage:
    solver = method(vrp_problem, **kwargs)
    try:
        solver.init_model(**kwargs)
    except:
        pass
    return solver.solve(**kwargs)


def return_solver(
    method: Type[SolverVrp], vrp_problem: VrpProblem, **kwargs: Any
) -> SolverVrp:
    solver = method(vrp_problem, **kwargs)
    try:
        solver.init_model(**kwargs)
    except:
        pass
    return solver
