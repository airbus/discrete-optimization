"""Utility module to launch different solvers on the coloring problem."""

#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any

from discrete_optimization.coloring.solvers.coloring_asp_solver import ColoringASPSolver
from discrete_optimization.coloring.solvers.coloring_cp_solvers import ColoringCP
from discrete_optimization.coloring.solvers.coloring_cpsat_solver import (
    ColoringCPSatSolver,
    ModelingCPSat,
)
from discrete_optimization.coloring.solvers.coloring_lp_solvers import (
    ColoringLP,
    ColoringLPMathOpt,
)
from discrete_optimization.coloring.solvers.coloring_solver import SolverColoring
from discrete_optimization.coloring.solvers.greedy_coloring import (
    ColoringProblem,
    GreedyColoring,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.generic_tools.do_problem import Problem
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

try:
    import pytoulbar2
except:
    toulbar2_available = False
else:
    from discrete_optimization.coloring.solvers.coloring_toulbar_solver import (
        ToulbarColoringSolver,
    )

    toulbar2_available = True


solvers: dict[str, list[tuple[type[SolverColoring], dict[str, Any]]]] = {
    "lp": [
        (
            ColoringLP,
            {},
        ),
        (
            ColoringLPMathOpt,
            {},
        ),
    ],
    "cp": [
        (
            ColoringCPSatSolver,
            {"modeling": ModelingCPSat.BINARY, "parameters_cp": ParametersCP.default()},
        ),
        (
            ColoringCP,
            {},
        ),
    ],
    "greedy": [(GreedyColoring, {})],
    "asp": [(ColoringASPSolver, {"time_limit": 5})],
}
if toulbar2_available:
    solvers["toulbar2"] = [(ToulbarColoringSolver, {"time_limit": 5})]

solvers_map = {}
for key in solvers:
    for solver, param in solvers[key]:
        solvers_map[solver] = (key, param)

solvers_compatibility: dict[type[SolverColoring], list[type[Problem]]] = {}
for x in solvers:
    for y in solvers[x]:
        solvers_compatibility[y[0]] = [ColoringProblem]


def look_for_solver(domain: "ColoringProblem") -> list[type[SolverColoring]]:
    """Given an instance of ColoringProblem, return a list of class of solvers.


    Args:
        domain (ColoringProblem): coloring problem instance

    Returns: list of solvers class
    """
    class_domain = domain.__class__
    return look_for_solver_class(class_domain)


def look_for_solver_class(
    class_domain: type[ColoringProblem],
) -> list[type[SolverColoring]]:
    """Given a class domain, return a list of class of solvers.


    Args:
        class_domain: should be ColoringProblem

    Returns: list of solvers class
    """
    available = []
    for solver in solvers_compatibility:
        if class_domain in solvers_compatibility[solver]:
            available += [solver]
    return available


def solve(
    method: type[SolverColoring], problem: ColoringProblem, **kwargs: Any
) -> ResultStorage:
    """Solve a coloring instance with a given class of solver.

    Args:
        method: class of the solver to use
        problem: coloring problem instance
        **args: specific options of the solver

    Returns: a ResultsStorage objecting obtained by the solver.

    """
    solver_ = method(problem, **kwargs)
    try:
        solver_.init_model(**kwargs)
    except AttributeError:
        pass
    return solver_.solve(**kwargs)


def return_solver(
    method: type[SolverColoring], problem: ColoringProblem, **kwargs: Any
) -> SolverColoring:
    """Return the solver initialized with the coloring problem instance

    Args:
        method: class of the solver to use
        problem: coloring problem instance
        **args: specific options of the solver

    Returns (SolverDO) : a solver object.

    """
    solver_ = method(problem, **kwargs)
    try:
        solver_.init_model(**kwargs)
    except AttributeError:
        pass
    return solver_
