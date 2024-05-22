"""Utility module to launch different solvers on the coloring problem."""

#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Tuple, Type

from discrete_optimization.coloring.solvers.coloring_cpsat_solver import ModelingCPSat
from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.generic_tools.lp_tools import ParametersMilp
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.maximum_independent_set.mis_model import MisProblem
from discrete_optimization.maximum_independent_set.solvers.mis_gurobi import (
    MisMilpSolver,
)
from discrete_optimization.maximum_independent_set.solvers.mis_ortools import (
    MisOrtoolsSolver,
)
from discrete_optimization.maximum_independent_set.solvers.mis_solver import MisSolver

solvers: Dict[str, List[Tuple[Type[MisSolver], Dict[str, Any]]]] = {
    "lp": [
        (
            MisMilpSolver,
            {
                "parameters_milp": ParametersMilp.default(),
            },
        ),
    ],
    "ortools": [
        (
            MisOrtoolsSolver,
            {"modeling": ModelingCPSat.BINARY, "parameters_cp": ParametersCP.default()},
        ),
    ],
}

solvers_map = {}
for key in solvers:
    for solver, param in solvers[key]:
        solvers_map[solver] = (key, param)


def solve(method: Type[MisSolver], problem: MisProblem, **kwargs: Any) -> ResultStorage:
    """Solve a mis instance with a given class of solver.

    Args:
        method: class of the solver to use
        problem: mis problem instance
        **args: specific options of the solver

    Returns: a ResultsStorage objecting obtained by the solver.

    """
    solver_ = method(problem, **kwargs)
    try:
        solver_.init_model(**kwargs)
    except AttributeError:
        pass
    return solver_.solve(**kwargs)
