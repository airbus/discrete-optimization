"""Utility module to launch different solvers on the coloring problem."""

#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Tuple, Type, Union

from discrete_optimization.coloring.solvers.coloring_quantum import (
    QAOAColoringSolver_FeasibleNbColor,
    QAOAColoringSolver_MinimizeNbColor,
    VQEColoringSolver_FeasibleNbColor,
    VQEColoringSolver_MinimizeNbColor,
)
from discrete_optimization.coloring.solvers.greedy_coloring import ColoringProblem
from discrete_optimization.generic_tools.qiskit_tools import QiskitSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.knapsack.knapsack_model import KnapsackModel
from discrete_optimization.knapsack.solvers.knapsack_quantum import (
    QAOAKnapsackSolver,
    VQEKnapsackSolver,
)
from discrete_optimization.maximum_independent_set.mis_model import MisProblem
from discrete_optimization.maximum_independent_set.solvers.mis_quantum import (
    QAOAMisSolver,
    VQEMisSolver,
)
from discrete_optimization.tsp.solver.tsp_quantum import QAOATSPSolver, VQETSPSolver
from discrete_optimization.tsp.tsp_model import TSPModel2D

solvers_coloring: Dict[str, List[Tuple[Type[QiskitSolver], Dict[str, Any]]]] = {
    "qaoa": [
        (
            QAOAColoringSolver_MinimizeNbColor,
            {},
        ),
        (
            QAOAColoringSolver_FeasibleNbColor,
            {},
        ),
    ],
    "vqe": [
        (
            VQEColoringSolver_MinimizeNbColor,
            {},
        ),
        (
            VQEColoringSolver_FeasibleNbColor,
            {},
        ),
    ],
}

solvers_map_coloring = {}
for key in solvers_coloring:
    for solver, param in solvers_coloring[key]:
        solvers_map_coloring[solver] = (key, param)


solvers_mis: Dict[str, List[Tuple[Type[QiskitSolver], Dict[str, Any]]]] = {
    "qaoa": [
        (
            QAOAMisSolver,
            {},
        ),
    ],
    "vqe": [
        (
            VQEMisSolver,
            {},
        ),
    ],
}

solvers_map_mis = {}
for key in solvers_mis:
    for solver, param in solvers_mis[key]:
        solvers_map_mis[solver] = (key, param)


solvers_tsp: Dict[str, List[Tuple[Type[QiskitSolver], Dict[str, Any]]]] = {
    "qaoa": [
        (
            QAOATSPSolver,
            {},
        ),
    ],
    "vqe": [
        (
            VQETSPSolver,
            {},
        ),
    ],
}

solvers_map_tsp = {}
for key in solvers_mis:
    for solver, param in solvers_tsp[key]:
        solvers_map_mis[solver] = (key, param)

solvers_knapsack: Dict[str, List[Tuple[Type[QiskitSolver], Dict[str, Any]]]] = {
    "qaoa": [
        (
            QAOAKnapsackSolver,
            {},
        ),
    ],
    "vqe": [
        (
            VQEKnapsackSolver,
            {},
        ),
    ],
}

solvers_map_knapsack = {}
for key in solvers_mis:
    for solver, param in solvers_knapsack[key]:
        solvers_map_mis[solver] = (key, param)


def solve(
    method: Type[QiskitSolver],
    problem: Union[MisProblem, TSPModel2D, KnapsackModel],
    **kwargs: Any
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

        solver_.init_model()
    except AttributeError:
        pass
    return solver_.solve(**kwargs)


def solve_coloring(
    method: Type[QiskitSolver], problem: ColoringProblem, nb_color, **kwargs: Any
) -> ResultStorage:
    """Solve a problem instance with a given class of solver.

    Args:
        method: class of the solver to use
        problem: problem instance
        nb_color: the number of colors or the max number of colors
        **args: specific options of the solver

    Returns: a ResultsStorage objecting obtained by the solver.

    """
    solver_ = method(problem, nb_color)
    try:

        solver_.init_model()
    except AttributeError:
        pass
    return solver_.solve(**kwargs)
