#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Tuple, Type

from discrete_optimization.facility.facility_model import (
    FacilityProblem,
    FacilityProblem2DPoints,
)
from discrete_optimization.facility.solvers.facility_cp_solvers import (
    CPSolverName,
    FacilityCP,
    FacilityCPModel,
    ParametersCP,
)
from discrete_optimization.facility.solvers.facility_lp_solver import (
    LP_Facility_Solver,
    LP_Facility_Solver_CBC,
    LP_Facility_Solver_PyMip,
    MilpSolverName,
    ParametersMilp,
)
from discrete_optimization.facility.solvers.facility_solver import SolverFacility
from discrete_optimization.facility.solvers.greedy_solvers import (
    GreedySolverDistanceBased,
    GreedySolverFacility,
)
from discrete_optimization.generic_tools.do_problem import Problem
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

solvers: Dict[str, List[Tuple[Type[SolverFacility], Dict[str, Any]]]] = {
    "lp": [
        (
            LP_Facility_Solver,
            {
                "parameters_milp": ParametersMilp.default(),
                "use_matrix_indicator_heuristic": True,
                "n_shortest": 10,
                "n_cheapest": 10,
            },
        ),
        (
            LP_Facility_Solver_CBC,
            {
                "parameters_milp": ParametersMilp.default(),
                "use_matrix_indicator_heuristic": True,
                "n_shortest": 10,
                "n_cheapest": 10,
            },
        ),
        (
            LP_Facility_Solver_PyMip,
            {
                "parameters_milp": ParametersMilp.default(),
                "use_matrix_indicator_heuristic": True,
                "milp_solver_name": MilpSolverName.CBC,
                "n_shortest": 10,
                "n_cheapest": 10,
            },
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

solvers_compatibility: Dict[Type[SolverFacility], List[Type[Problem]]] = {}
for x in solvers:
    for y in solvers[x]:
        solvers_compatibility[y[0]] = [FacilityProblem2DPoints]


def look_for_solver(domain: FacilityProblem) -> List[Type[SolverFacility]]:
    class_domain = domain.__class__
    return look_for_solver_class(class_domain)


def look_for_solver_class(
    class_domain: Type[FacilityProblem],
) -> List[Type[SolverFacility]]:
    available = []
    for solver in solvers_compatibility:
        if class_domain in solvers_compatibility[solver]:
            available += [solver]
    return available


def solve(
    method: Type[SolverFacility], facility_problem: FacilityProblem, **kwargs: Any
) -> ResultStorage:
    solver = method(facility_problem, **kwargs)
    try:
        solver.init_model(**kwargs)
    except:
        pass
    return solver.solve(**kwargs)


def return_solver(
    method: Type[SolverFacility], coloring_model: FacilityProblem, **kwargs: Any
) -> SolverFacility:
    solver = method(coloring_model, **kwargs)
    try:
        solver.init_model(**kwargs)
    except:
        pass
    return solver
