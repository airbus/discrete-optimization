#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.vrp.solver.lp_vrp_iterative import VRPIterativeLP
from discrete_optimization.vrp.solver.lp_vrp_iterative_pymip import VRPIterativeLP_Pymip
from discrete_optimization.vrp.solver.solver_ortools import VrpORToolsSolver
from discrete_optimization.vrp.vrp_model import VrpProblem, VrpProblem2D

solvers = {
    "ortools": [(VrpORToolsSolver, {"limit_time_s": 100})],
    "lp": [(VRPIterativeLP, {}), (VRPIterativeLP_Pymip, {})],
}

solvers_map = {}
for key in solvers:
    for solver, param in solvers[key]:
        solvers_map[solver] = (key, param)

solvers_compatibility = {}
for x in solvers:
    for y in solvers[x]:
        solvers_compatibility[y[0]] = [VrpProblem, VrpProblem2D]


def look_for_solver(domain):
    class_domain = domain.__class__
    return look_for_solver_class(class_domain)


def look_for_solver_class(class_domain):
    available = []
    for solver in solvers_compatibility:
        if class_domain in solvers_compatibility[solver]:
            available += [solver]
    return available


def solve(method, vrp_problem: VrpProblem, **args) -> ResultStorage:
    solver = method(vrp_problem, **args)
    try:
        solver.init_model(**args)
    except:
        pass
    return solver.solve(**args)


def return_solver(method, vrp_problem: VrpProblem, **args) -> ResultStorage:
    solver = method(vrp_problem, **args)
    try:
        solver.init_model(**args)
    except:
        pass
    return solver
