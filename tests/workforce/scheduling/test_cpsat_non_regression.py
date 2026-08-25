#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from pytest_cases import parametrize_with_cases

NONREGRESSION_POPULATE_DATABASE = False
"""Whether to create the database or use it to test against it.

- True: we create the non-regression database
- False: we compare the results with the ones previously stored with the non-regression database

"""


logger = logging.getLogger(__name__)


@parametrize_with_cases("problem", prefix="problem_")
@parametrize_with_cases(
    "solver_cls, solver_kwargs, objective_fn, mode_optim", prefix="solver_"
)
def test_cpsat_solver(
    request,
    problem,
    solver_cls,
    solver_kwargs,
    objective_fn,
    mode_optim,
    check_nonregression_fn,
):
    solver = solver_cls(problem=problem, **solver_kwargs)
    solver.init_model(**solver_kwargs)
    res = solver.solve(**solver_kwargs)
    sol, fit = res[-1]
    objective = objective_fn(sol)
    test_id = request.node.nodeid
    status = solver.status_solver
    # compare with previous runs (or populate the nonregression database)
    check_nonregression_fn(
        test_id=test_id, status=status, objective=objective, mode_optim=mode_optim
    )
