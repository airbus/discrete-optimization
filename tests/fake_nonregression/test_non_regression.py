#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from pytest_cases import parametrize

from discrete_optimization.generic_tools.do_problem import ModeOptim
from discrete_optimization.generic_tools.do_solver import StatusSolver

objectives = [10, (1, 3), (1, 3.5), (1, 2), 25]

mode_optims = [
    ModeOptim.MINIMIZATION,
    ModeOptim.MINIMIZATION,
    ModeOptim.MINIMIZATION,
    ModeOptim.MAXIMIZATION,
    ModeOptim.MAXIMIZATION,
]

statuses = [
    StatusSolver.OPTIMAL,
    StatusSolver.SATISFIED,
    StatusSolver.SATISFIED,
    StatusSolver.SATISFIED,
    StatusSolver.SATISFIED,
]
NONREGRESSION_POPULATE_DATABASE = False


from discrete_optimization.toto.b import B


@parametrize("test_id", [0, 1, 2, 3, 4])
def test_cpsat(test_id, check_nonregression_fn):
    B(test_id)
    check_nonregression_fn(
        test_id=test_id,
        objective=objectives[test_id],
        status=statuses[test_id],
        mode_optim=mode_optims[test_id],
    )
