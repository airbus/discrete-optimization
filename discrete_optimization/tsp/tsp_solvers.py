#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Sequence, Tuple, Type

from discrete_optimization.generic_tools.do_problem import Problem
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.tsp.solver.solver_lp_iterative import (
    LP_TSP_Iterative,
    MILPSolver,
)
from discrete_optimization.tsp.solver.solver_ortools import TSP_ORtools
from discrete_optimization.tsp.solver.tsp_cp_solver import (
    CPSolverName,
    TSP_CP_Solver,
    TSP_CPModel,
)
from discrete_optimization.tsp.solver.tsp_solver import SolverTSP
from discrete_optimization.tsp.tsp_model import (
    TSPModel,
    TSPModel2D,
    TSPModelDistanceMatrix,
)

solvers: Dict[str, List[Tuple[Type[SolverTSP], Dict[str, Any]]]] = {
    "lp": [
        (
            LP_TSP_Iterative,
            {"method": MILPSolver.CBC, "nb_iteration_max": 20, "plot": False},
        )
    ],
    "ortools": [(TSP_ORtools, {})],
    "cp": [
        (
            TSP_CP_Solver,
            {
                "model_type": TSP_CPModel.INT_VERSION,
                "cp_solver_name": CPSolverName.CHUFFED,
            },
        )
    ],
}

solvers_map = {}
for key in solvers:
    for solver, param in solvers[key]:
        solvers_map[solver] = (key, param)

solvers_compatibility: Dict[Type[SolverTSP], List[Type[Problem]]] = {}
for x in solvers:
    for y in solvers[x]:
        solvers_compatibility[y[0]] = [TSPModel2D, TSPModelDistanceMatrix]


def look_for_solver(domain: TSPModel) -> List[Type[SolverTSP]]:
    class_domain = domain.__class__
    return look_for_solver_class(class_domain)


def look_for_solver_class(class_domain: Type["TSPModel"]) -> List[Type[SolverTSP]]:
    available = []
    for solver in solvers_compatibility:
        if class_domain in solvers_compatibility[solver]:
            available += [solver]
    return available


def solve(
    method: Type[SolverTSP], tsp_problem: TSPModel, **kwargs: Any
) -> ResultStorage:
    solver = method(tsp_model=tsp_problem, **kwargs)
    try:
        solver.init_model(**kwargs)
    except AttributeError:
        pass
    return solver.solve(**kwargs)


def return_solver(
    method: Type[SolverTSP], tsp_problem: TSPModel, **kwargs: Any
) -> SolverTSP:
    solver = method(tsp_problem, **kwargs)
    try:
        solver.init_model(**kwargs)
    except:
        pass
    return solver
