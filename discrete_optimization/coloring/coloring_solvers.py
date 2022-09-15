"""Utility module to launch different solvers on the coloring problem."""
from typing import Any, Dict, List, Tuple, Type

from discrete_optimization.coloring.solvers.coloring_cp_solvers import (
    ColoringCP,
    ColoringCPModel,
    CPSolverName,
    ParametersCP,
)
from discrete_optimization.coloring.solvers.coloring_lp_solvers import (
    ColoringLP,
    ColoringLP_MIP,
    MilpSolverName,
)
from discrete_optimization.coloring.solvers.greedy_coloring import (
    ColoringProblem,
    GreedyColoring,
    NXGreedyColoringMethod,
)
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.lp_tools import ParametersMilp
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

solvers: Dict[str, List[Tuple[Type, Dict[str, Any]]]] = {
    "lp": [
        (
            ColoringLP,
            {
                "greedy_start": True,
                "use_cliques": False,
                "parameters_milp": ParametersMilp.default(),
            },
        ),
        (
            ColoringLP_MIP,
            {
                "milp_solver_name": MilpSolverName.CBC,
                "greedy_start": True,
                "parameters_milp": ParametersMilp.default(),
                "use_cliques": False,
            },
        ),
    ],
    "cp": [
        (
            ColoringCP,
            {
                "cp_solver_name": CPSolverName.CHUFFED,
                "cp_model": ColoringCPModel.DEFAULT,
                "parameters_cp": ParametersCP.default(),
                "object_output": True,
            },
        )
    ],
    "greedy": [(GreedyColoring, {"strategy": NXGreedyColoringMethod.best})],
}

solvers_map = {}
for key in solvers:
    for solver, param in solvers[key]:
        solvers_map[solver] = (key, param)

solvers_compatibility = {}
for x in solvers:
    for y in solvers[x]:
        solvers_compatibility[y[0]] = [ColoringProblem]


def look_for_solver(domain: "ColoringProblem"):
    """Given an instance of ColoringProblem, return a list of class of solvers.


    Args:
        domain (ColoringProblem): coloring problem instance

    Returns: list of solvers class
    """
    class_domain = domain.__class__
    return look_for_solver_class(class_domain)


def look_for_solver_class(class_domain):
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


def solve(method, coloring_model: ColoringProblem, **args) -> ResultStorage:
    """Solve a coloring instance with a given class of solver.

    Args:
        method: class of the solver to use
        coloring_model: coloring problem instance
        **args: specific options of the solver

    Returns: a ResultsStorage objecting obtained by the solver.

    """
    solver_ = method(coloring_model, **args)
    try:
        solver_.init_model(**args)
    except AttributeError:
        pass
    return solver_.solve(**args)


def return_solver(method, coloring_model: ColoringProblem, **args) -> SolverDO:
    """Return the solver initialized with the coloring problem instance

    Args:
        method: class of the solver to use
        coloring_model: coloring problem instance
        **args: specific options of the solver

    Returns (SolverDO) : a solver object.

    """
    solver_ = method(coloring_model, **args)
    try:
        solver_.init_model(**args)
    except AttributeError:
        pass
    return solver_
