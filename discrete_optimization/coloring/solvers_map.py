"""Utility module to launch different solvers on the coloring problem."""

#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any

from discrete_optimization.coloring.solvers import ColoringSolver
from discrete_optimization.coloring.solvers.asp import AspColoringSolver
from discrete_optimization.coloring.solvers.cp_mzn import CpColoringSolver
from discrete_optimization.coloring.solvers.cpsat import (
    CpSatColoringSolver,
    ModelingCpSat,
)
from discrete_optimization.coloring.solvers.greedy import (
    ColoringProblem,
    GreedyColoringSolver,
)
from discrete_optimization.coloring.solvers.lp import (
    GurobiColoringSolver,
    MathOptColoringSolver,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_problem import Problem
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

try:
    import pytoulbar2
except:
    toulbar2_available = False
else:
    from discrete_optimization.coloring.solvers.toulbar import ToulbarColoringSolver

    toulbar2_available = True


solvers: dict[str, list[tuple[type[ColoringSolver], dict[str, Any]]]] = {
    "lp": [
        (
            GurobiColoringSolver,
            {},
        ),
        (
            MathOptColoringSolver,
            {},
        ),
    ],
    "cp": [
        (
            CpSatColoringSolver,
            {"modeling": ModelingCpSat.BINARY, "parameters_cp": ParametersCp.default()},
        ),
        (
            CpColoringSolver,
            {},
        ),
    ],
    "greedy": [(GreedyColoringSolver, {})],
    "asp": [(AspColoringSolver, {"time_limit": 5})],
}
if toulbar2_available:
    solvers["toulbar2"] = [(ToulbarColoringSolver, {"time_limit": 5})]

solvers_map = {}
for key in solvers:
    for solver, param in solvers[key]:
        solvers_map[solver] = (key, param)

solvers_compatibility: dict[type[ColoringSolver], list[type[Problem]]] = {}
for x in solvers:
    for y in solvers[x]:
        solvers_compatibility[y[0]] = [ColoringProblem]


def look_for_solver(domain: "ColoringProblem") -> list[type[ColoringSolver]]:
    """Given an instance of ColoringProblem, return a list of class of solvers.


    Args:
        domain (ColoringProblem): coloring problem instance

    Returns: list of solvers class
    """
    class_domain = domain.__class__
    return look_for_solver_class(class_domain)


def look_for_solver_class(
    class_domain: type[ColoringProblem],
) -> list[type[ColoringSolver]]:
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
    method: type[ColoringSolver], problem: ColoringProblem, **kwargs: Any
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
    method: type[ColoringSolver], problem: ColoringProblem, **kwargs: Any
) -> ColoringSolver:
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
