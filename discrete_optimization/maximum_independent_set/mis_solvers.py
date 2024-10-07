"""Utility module to launch different solvers on the maximum independent set problem."""

#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any

from discrete_optimization.generic_tools.lp_tools import ParametersMilp
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.maximum_independent_set.mis_model import MisProblem
from discrete_optimization.maximum_independent_set.solvers.mis_asp import MisASPSolver
from discrete_optimization.maximum_independent_set.solvers.mis_gurobi import (
    MisMilpSolver,
    MisQuadraticSolver,
)
from discrete_optimization.maximum_independent_set.solvers.mis_kamis import (
    MisKamisSolver,
)
from discrete_optimization.maximum_independent_set.solvers.mis_networkx import (
    MisNetworkXSolver,
)
from discrete_optimization.maximum_independent_set.solvers.mis_ortools import (
    MisOrtoolsSolver,
)
from discrete_optimization.maximum_independent_set.solvers.mis_solver import MisSolver
from discrete_optimization.maximum_independent_set.solvers.mis_toulbar import (
    MisToulbarSolver,
)

solvers: dict[str, list[tuple[type[MisSolver], dict[str, Any]]]] = {
    "lp": [
        (
            MisMilpSolver,
            {
                "parameters_milp": ParametersMilp.default(),
            },
        ),
        (
            MisQuadraticSolver,
            {
                "parameters_milp": ParametersMilp.default(),
            },
        ),
    ],
    "cp": [
        (
            MisOrtoolsSolver,
            {},
        ),
    ],
    "networkX": [(MisNetworkXSolver, {})],
    "kamis": [(MisKamisSolver, {})],
    "asp": [(MisASPSolver, {"time_limit": 20})],
    "toulbar": [(MisToulbarSolver, {"time_limit": 20})],
}

solvers_map = {}
for key in solvers:
    for solver, param in solvers[key]:
        solvers_map[solver] = (key, param)


def solve(
    method_solver: type[MisSolver], problem: MisProblem, **kwargs: Any
) -> ResultStorage:
    """Solve a mis instance with a given class of solver.

    Args:
        method_solver: class of the solver to use
        problem: mis problem instance
        **args: specific options of the solver

    Returns: a ResultsStorage objecting obtained by the solver.

    """
    solver_ = method_solver(problem, **kwargs)
    try:
        solver_.init_model(**kwargs)
    except AttributeError:
        pass
    return solver_.solve(**kwargs)
