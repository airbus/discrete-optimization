#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.transformation.composite import (
    CompositeTransformation,
)
from discrete_optimization.generic_tools.transformation.transformation_solver import (
    SubBrick,
    TransformationSolver,
)
from discrete_optimization.shop.fjsp.transformations.to_rcpsp import (
    FjspToRcpspTransformation,
)
from discrete_optimization.shop.jsp.parser import get_data_available, parse_file
from discrete_optimization.shop.jsp.problem import JobShopProblem
from discrete_optimization.shop.jsp.transformations.to_fjsp import (
    JspToFjspTransformation,
)


@pytest.fixture()
def problem() -> JobShopProblem:
    filename = "la02"
    filepath = [f for f in get_data_available() if f.endswith(filename)][0]
    return parse_file(filepath)


def test_via_fjsp(problem):
    from discrete_optimization.shop.fjsp.solvers.cpsat_auto import CpSatAutoFjspSolver

    solver = TransformationSolver(
        transformation=JspToFjspTransformation(),
        source_problem=problem,
        solver_brick=SubBrick(CpSatAutoFjspSolver, {}),
    )
    res = solver.solve(callbacks=[NbIterationStopper(1)])
    sol = res[-1][0]
    assert problem.satisfy(sol)


def test_via_rcpsp(problem):
    from discrete_optimization.rcpsp.solvers.pile import PileRcpspSolver

    solver = TransformationSolver(
        transformation=CompositeTransformation(
            [JspToFjspTransformation(), FjspToRcpspTransformation()]
        ),
        source_problem=problem,
        solver_brick=SubBrick(PileRcpspSolver, {}),
    )
    res = solver.solve(callbacks=[NbIterationStopper(1)])
    sol = res[-1][0]
    assert problem.satisfy(sol)
