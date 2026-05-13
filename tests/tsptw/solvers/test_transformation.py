#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import discrete_optimization.tsptw.parser as tsptw_parser
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.transformation import TransformationSolver
from discrete_optimization.tsptw.transformations.to_vrptw import (
    TsptwToVrptwTransformation,
)
from discrete_optimization.vrptw.solvers.cpsat import CpSatVRPTWSolver


def test_via_vrptw():
    """Solve TSPTW via VRPTW transformation."""
    files = tsptw_parser.get_data_available()
    print(files)
    file = [f for f in files if "rc_201.1.txt" in f][0]
    problem = tsptw_parser.parse_tsptw_file(file)
    solver = TransformationSolver(
        transformation=TsptwToVrptwTransformation(),
        source_problem=problem,
        solver_brick=SubBrick(cls=CpSatVRPTWSolver, kwargs={}),
    )
    res = solver.solve(callbacks=[NbIterationStopper(1)], time_limit=10)
    sol = res[-1][0]
    print(f"Evaluate: {problem.evaluate(sol)}, Satisfy: {problem.satisfy(sol)}")
