"""Utility module to launch different solvers on the coloring problem."""

#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any, Union

from discrete_optimization.coloring.solvers.greedy import ColoringProblem
from discrete_optimization.coloring.solvers.quantum import (
    FeasibleNbColorQaoaColoringSolver,
    FeasibleNbColorVqeColoringSolver,
    MinimizeNbColorQaoaColoringSolver,
    MinimizeNbColorVqeColoringSolver,
)
from discrete_optimization.facility.problem import Facility2DProblem
from discrete_optimization.facility.solvers.quantum import (
    QaoaFacilitySolver,
    VqeFacilitySolver,
)
from discrete_optimization.generic_tools.qiskit_tools import QiskitSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.knapsack.problem import KnapsackProblem
from discrete_optimization.knapsack.solvers.quantum import (
    QaoaKnapsackSolver,
    VqeKnapsackSolver,
)
from discrete_optimization.maximum_independent_set.problem import MisProblem
from discrete_optimization.maximum_independent_set.solvers.quantum import (
    QaoaMisSolver,
    VqeMisSolver,
)
from discrete_optimization.tsp.problem import Point2DTspProblem
from discrete_optimization.tsp.solvers.quantum import QaoaTspSolver, VqeTspSolver

solvers_coloring: dict[str, list[tuple[type[QiskitSolver], dict[str, Any]]]] = {
    "qaoa": [
        (
            MinimizeNbColorQaoaColoringSolver,
            {},
        ),
        (
            FeasibleNbColorQaoaColoringSolver,
            {},
        ),
    ],
    "vqe": [
        (
            MinimizeNbColorVqeColoringSolver,
            {},
        ),
        (
            FeasibleNbColorVqeColoringSolver,
            {},
        ),
    ],
}

solvers_map_coloring = {}
for key, solver_configs in solvers_coloring.items():
    for solver, param in solver_configs:
        solvers_map_coloring[solver] = (key, param)


solvers_mis: dict[str, list[tuple[type[QiskitSolver], dict[str, Any]]]] = {
    "qaoa": [
        (
            QaoaMisSolver,
            {},
        ),
    ],
    "vqe": [
        (
            VqeMisSolver,
            {},
        ),
    ],
}

solvers_map_mis = {}
for key, solver_configs in solvers_mis.items():
    for solver, param in solver_configs:
        solvers_map_mis[solver] = (key, param)

solvers_facility: dict[str, list[tuple[type[QiskitSolver], dict[str, Any]]]] = {
    "qaoa": [
        (
            QaoaFacilitySolver,
            {},
        ),
    ],
    "vqe": [
        (
            VqeFacilitySolver,
            {},
        ),
    ],
}

solvers_map_facility = {}
for key, solver_configs in solvers_facility.items():
    for solver, param in solver_configs:
        solvers_map_facility[solver] = (key, param)


solvers_tsp: dict[str, list[tuple[type[QiskitSolver], dict[str, Any]]]] = {
    "qaoa": [
        (
            QaoaTspSolver,
            {},
        ),
    ],
    "vqe": [
        (
            VqeTspSolver,
            {},
        ),
    ],
}

solvers_map_tsp = {}
for key, solver_configs in solvers_tsp.items():
    for solver, param in solver_configs:
        solvers_map_tsp[solver] = (key, param)

solvers_knapsack: dict[str, list[tuple[type[QiskitSolver], dict[str, Any]]]] = {
    "qaoa": [
        (
            QaoaKnapsackSolver,
            {},
        ),
    ],
    "vqe": [
        (
            VqeKnapsackSolver,
            {},
        ),
    ],
}

solvers_map_knapsack = {}
for key, solver_configs in solvers_knapsack.items():
    for solver, param in solver_configs:
        solvers_map_knapsack[solver] = (key, param)


def solve(
    method: type[QiskitSolver],
    problem: Union[MisProblem, Point2DTspProblem, KnapsackProblem, Facility2DProblem],
    **kwargs: Any,
) -> ResultStorage:
    """Solve a problem instance with a given class of solver.

    Args:
        method: class of the solver to use
        problem: problem instance
        **args: specific options of the solver

    Returns: a ResultsStorage objecting obtained by the solver.

    """
    solver_ = method(problem, **kwargs)
    try:

        solver_.init_model(**kwargs)
    except AttributeError:
        pass
    return solver_.solve(**kwargs)


def solve_coloring(
    method: type[QiskitSolver], problem: ColoringProblem, nb_color, **kwargs: Any
) -> ResultStorage:
    """Solve a problem instance with a given class of solver.

    Args:
        method: class of the solver to use
        problem: problem instance
        nb_color: the number of colors or the max number of colors
        **args: specific options of the solver

    Returns: a ResultsStorage objecting obtained by the solver.

    """
    solver_ = method(problem, nb_color, **kwargs)
    try:

        solver_.init_model(**kwargs)
    except AttributeError:
        pass
    return solver_.solve(**kwargs)
