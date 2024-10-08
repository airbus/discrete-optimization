#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any

from discrete_optimization.facility.facility_model import (
    FacilityProblem,
    FacilityProblem2DPoints,
)
from discrete_optimization.facility.solvers.facility_cp_solvers import (
    FacilityCP,
    FacilityCPModel,
)
from discrete_optimization.facility.solvers.facility_lp_solver import (
    LP_Facility_Solver,
    LP_Facility_Solver_CBC,
    LP_Facility_Solver_MathOpt,
)
from discrete_optimization.facility.solvers.facility_solver import SolverFacility
from discrete_optimization.facility.solvers.greedy_solvers import (
    GreedySolverDistanceBased,
    GreedySolverFacility,
)
from discrete_optimization.generic_tools.cp_tools import CPSolverName, ParametersCP
from discrete_optimization.generic_tools.do_problem import Problem
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

solvers: dict[str, list[tuple[type[SolverFacility], dict[str, Any]]]] = {
    "lp": [
        (
            LP_Facility_Solver_MathOpt,
            {},
        ),
        (
            LP_Facility_Solver,
            {},
        ),
        (
            LP_Facility_Solver_CBC,
            {},
        ),
    ],
    "cp": [
        (
            FacilityCP,
            {
                "cp_solver_name": CPSolverName.CHUFFED,
                "object_output": True,
                "cp_model": FacilityCPModel.DEFAULT_INT,
                "parameters_cp": ParametersCP.default(),
            },
        )
    ],
    "greedy": [(GreedySolverFacility, {}), (GreedySolverDistanceBased, {})],
}

solvers_map = {}
for key in solvers:
    for solver, param in solvers[key]:
        solvers_map[solver] = (key, param)

solvers_compatibility: dict[type[SolverFacility], list[type[Problem]]] = {}
for x in solvers:
    for y in solvers[x]:
        solvers_compatibility[y[0]] = [FacilityProblem2DPoints]


def look_for_solver(domain: FacilityProblem) -> list[type[SolverFacility]]:
    class_domain = domain.__class__
    return look_for_solver_class(class_domain)


def look_for_solver_class(
    class_domain: type[FacilityProblem],
) -> list[type[SolverFacility]]:
    available = []
    for solver in solvers_compatibility:
        if class_domain in solvers_compatibility[solver]:
            available += [solver]
    return available


def solve(
    method: type[SolverFacility], problem: FacilityProblem, **kwargs: Any
) -> ResultStorage:
    solver = method(problem, **kwargs)
    try:
        solver.init_model(**kwargs)
    except:
        pass
    return solver.solve(**kwargs)


def return_solver(
    method: type[SolverFacility], problem: FacilityProblem, **kwargs: Any
) -> SolverFacility:
    solver = method(problem, **kwargs)
    try:
        solver.init_model(**kwargs)
    except:
        pass
    return solver
