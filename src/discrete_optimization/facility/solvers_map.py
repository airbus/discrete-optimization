#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any

from discrete_optimization.facility.problem import Facility2DProblem, FacilityProblem
from discrete_optimization.facility.solvers import FacilitySolver
from discrete_optimization.facility.solvers.cp_mzn import (
    CpFacilityModel,
    CpFacilitySolver,
)
from discrete_optimization.facility.solvers.greedy import (
    DistanceBasedGreedyFacilitySolver,
    GreedyFacilitySolver,
)
from discrete_optimization.facility.solvers.lp import (
    CbcFacilitySolver,
    GurobiFacilitySolver,
    MathOptFacilitySolver,
)
from discrete_optimization.generic_tools.cp_tools import CpSolverName, ParametersCp
from discrete_optimization.generic_tools.do_problem import Problem
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

solvers: dict[str, list[tuple[type[FacilitySolver], dict[str, Any]]]] = {
    "lp": [
        (
            MathOptFacilitySolver,
            {},
        ),
        (
            GurobiFacilitySolver,
            {},
        ),
        (
            CbcFacilitySolver,
            {},
        ),
    ],
    "cp": [
        (
            CpFacilitySolver,
            {
                "cp_solver_name": CpSolverName.CHUFFED,
                "object_output": True,
                "cp_model": CpFacilityModel.DEFAULT_INT,
                "parameters_cp": ParametersCp.default(),
            },
        )
    ],
    "greedy": [(GreedyFacilitySolver, {}), (DistanceBasedGreedyFacilitySolver, {})],
}

solvers_map = {}
for key in solvers:
    for solver, param in solvers[key]:
        solvers_map[solver] = (key, param)

solvers_compatibility: dict[type[FacilitySolver], list[type[Problem]]] = {}
for x in solvers:
    for y in solvers[x]:
        solvers_compatibility[y[0]] = [Facility2DProblem]


def look_for_solver(domain: FacilityProblem) -> list[type[FacilitySolver]]:
    class_domain = domain.__class__
    return look_for_solver_class(class_domain)


def look_for_solver_class(
    class_domain: type[FacilityProblem],
) -> list[type[FacilitySolver]]:
    available = []
    for solver in solvers_compatibility:
        if class_domain in solvers_compatibility[solver]:
            available += [solver]
    return available


def solve(
    method: type[FacilitySolver], problem: FacilityProblem, **kwargs: Any
) -> ResultStorage:
    solver = method(problem, **kwargs)
    try:
        solver.init_model(**kwargs)
    except:
        pass
    return solver.solve(**kwargs)


def return_solver(
    method: type[FacilitySolver], problem: FacilityProblem, **kwargs: Any
) -> FacilitySolver:
    solver = method(problem, **kwargs)
    try:
        solver.init_model(**kwargs)
    except:
        pass
    return solver
