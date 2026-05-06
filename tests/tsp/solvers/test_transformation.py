#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import discrete_optimization.tsp.parser as tsp_parser
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.transformation import TransformationSolver
from discrete_optimization.tsp.transformations.to_tsptw import (
    TspToTsptwTransformation,
)
from discrete_optimization.tsp.transformations.to_vrp import TspToVrpTransformation
from discrete_optimization.tsptw.solvers.cpsat import CpSatTSPTWSolver
from discrete_optimization.vrp.solvers.cpsat import CpSatVrpSolver


def test_via_tsptw():
    """Solve TSP via TSPTW transformation."""
    files = tsp_parser.get_data_available()
    print(files)
    file = [f for f in files if "tsp_5_1" in f][0]
    problem = tsp_parser.parse_file(file)
    solver = TransformationSolver(
        transformation=TspToTsptwTransformation(),
        source_problem=problem,
        solver_brick=SubBrick(cls=CpSatTSPTWSolver, kwargs={}),
    )
    res = solver.solve(callbacks=[NbIterationStopper(1)], time_limit=10)
    sol = res[-1][0]
    assert problem.satisfy(sol)


def test_via_vrp():
    """Solve TSP via VRP transformation."""
    files = tsp_parser.get_data_available()
    print(files)
    file = [f for f in files if "tsp_5_1" in f][0]
    problem = tsp_parser.parse_file(file)
    solver = TransformationSolver(
        transformation=TspToVrpTransformation(),
        source_problem=problem,
        solver_brick=SubBrick(cls=CpSatVrpSolver, kwargs={}),
    )
    res = solver.solve(callbacks=[NbIterationStopper(1)], time_limit=10)
    sol = res[-1][0]
    assert problem.satisfy(sol)
