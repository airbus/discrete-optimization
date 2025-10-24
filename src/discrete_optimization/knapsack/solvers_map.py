#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any

from discrete_optimization.generic_tools.do_problem import Problem
from discrete_optimization.generic_tools.lp_tools import ParametersMilp
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.knapsack.problem import KnapsackProblem
from discrete_optimization.knapsack.solvers import KnapsackSolver
from discrete_optimization.knapsack.solvers.asp import AspKnapsackSolver
from discrete_optimization.knapsack.solvers.cp_mzn import (
    Cp2KnapsackSolver,
    CpKnapsackSolver,
)
from discrete_optimization.knapsack.solvers.cpsat import CpSatKnapsackSolver
from discrete_optimization.knapsack.solvers.dp import ExactDpKnapsackSolver
from discrete_optimization.knapsack.solvers.greedy import GreedyBestKnapsackSolver
from discrete_optimization.knapsack.solvers.lp import (
    CbcKnapsackSolver,
    GurobiKnapsackSolver,
    OrtoolsKnapsackSolver,
)

solvers: dict[str, list[tuple[type[KnapsackSolver], dict[str, Any]]]] = {
    "lp": [
        (OrtoolsKnapsackSolver, {}),
        (CbcKnapsackSolver, {}),
        (GurobiKnapsackSolver, {"parameter_gurobi": ParametersMilp.default()}),
    ],
    "greedy": [(GreedyBestKnapsackSolver, {})],
    "cp": [
        (CpSatKnapsackSolver, {}),
        (CpKnapsackSolver, {}),
        (Cp2KnapsackSolver, {}),
    ],
    "asp": [(AspKnapsackSolver, {})],
    "dyn_prog": [
        (
            ExactDpKnapsackSolver,
            {
                "greedy_start": True,
                "stop_after_n_item": True,
                "max_items": 100,
                "time_limit": 100,
            },
        )
    ],
}

solvers_map = {}
for key in solvers:
    for solver, param in solvers[key]:
        solvers_map[solver] = (key, param)

solvers_compatibility: dict[type[KnapsackSolver], list[type[Problem]]] = {}
for x in solvers:
    for y in solvers[x]:
        solvers_compatibility[y[0]] = [KnapsackProblem]


def look_for_solver(domain: KnapsackProblem) -> list[type[KnapsackSolver]]:
    class_domain = domain.__class__
    return look_for_solver_class(class_domain)


def look_for_solver_class(
    class_domain: type[KnapsackProblem],
) -> list[type[KnapsackSolver]]:
    available = []
    for solver in solvers_compatibility:
        if class_domain in solvers_compatibility[solver]:
            available += [solver]
    return available


def solve(
    method: type[KnapsackSolver], problem: KnapsackProblem, **args: Any
) -> ResultStorage:
    solver = method(problem, **args)
    solver.init_model(**args)
    return solver.solve(**args)
