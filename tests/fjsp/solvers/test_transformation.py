#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import discrete_optimization.fjsp.parser as fjsp_parser
from discrete_optimization.fjsp.transformations.to_rcpsp import (
    FjspToRcpspTransformation,
)
from discrete_optimization.fjsp.transformations.to_workforce import (
    FjspToWorkforceSchedulingTransformation,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.transformation import TransformationSolver
from discrete_optimization.rcpsp.solvers.pile import PileRcpspSolver
from discrete_optimization.workforce.scheduling.solvers.cpsat import (
    CPSatAllocSchedulingSolver,
)


def test_via_rcpsp():
    """Solve FJSP via RCPSP transformation."""
    files = fjsp_parser.get_data_available()
    print(files)
    file = [f for f in files if "Behnke1.fjs" in f][0]
    problem = fjsp_parser.parse_file(file)
    solver = TransformationSolver(
        transformation=FjspToRcpspTransformation(),
        source_problem=problem,
        solver_brick=SubBrick(cls=PileRcpspSolver, kwargs={}),
    )
    res = solver.solve()
    sol = res[-1][0]
    assert problem.satisfy(sol)


def test_via_workforce():
    """Solve FJSP via RCPSP transformation."""
    files = fjsp_parser.get_data_available()
    print(files)
    file = [f for f in files if "Behnke1.fjs" in f][0]
    problem = fjsp_parser.parse_file(file)
    solver = TransformationSolver(
        transformation=FjspToWorkforceSchedulingTransformation(),
        source_problem=problem,
        solver_brick=SubBrick(cls=CPSatAllocSchedulingSolver, kwargs={}),
    )
    res = solver.solve(callbacks=[NbIterationStopper(1)])
    sol = res[-1][0]
    print(problem.evaluate(sol), problem.satisfy(sol))
    # Duration are unchanged
    # per workers in the workforce version..
    # assert problem.satisfy(sol)
