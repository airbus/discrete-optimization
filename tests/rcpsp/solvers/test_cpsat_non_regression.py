#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from pytest_cases import parametrize_with_cases

from discrete_optimization.generic_tools.do_problem import ModeOptim
from discrete_optimization.generic_tools.do_solver import StatusSolver

NONREGRESSION_POPULATE_DATABASE = False
"""Whether to create the database or use it to test against it.

- True: we create the non-regression database
- False: we compare the results with the ones previously stored with the non-regression database

"""


@parametrize_with_cases("test_id, status, objective, mode_optim")
def test_rcpsp_solver(
    test_id,
    status,
    objective,
    mode_optim,
    check_nonregression_fn,
):
    if (
        test_id
        == "tests/rcpsp/solvers/test_cpsat_non_regression.py::test_rcpsp_solver[simple-j1010_1.mm-cpsat_resource-30]"
    ):
        status = StatusSolver.SATISFIED
    if (
        test_id
        == "tests/rcpsp/solvers/test_cpsat_non_regression.py::test_rcpsp_solver[simple-j301_1.sm-cpsat-30]"
    ):
        mode_optim = ModeOptim.MAXIMIZATION

    # compare with previous runs (or populate the nonregression database)
    check_nonregression_fn(
        test_id=test_id, status=status, objective=objective, mode_optim=mode_optim
    )
