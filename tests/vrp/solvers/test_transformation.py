#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import discrete_optimization.vrp.parser as vrp_parser
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.transformation import TransformationSolver
from discrete_optimization.vrp.transformations.to_vrptw import VrpToVrptwTransformation
from discrete_optimization.vrptw.solvers.cpsat import CpSatVRPTWSolver


def test_via_vrptw():
    """Solve VRP via VRPTW transformation."""
    files = vrp_parser.get_data_available()
    print(files)
    file = [f for f in files if "vrp_16_3_1" in f][0]
    problem = vrp_parser.parse_file(file)
    solver = TransformationSolver(
        transformation=VrpToVrptwTransformation(),
        source_problem=problem,
        solver_brick=SubBrick(cls=CpSatVRPTWSolver, kwargs={}),
    )
    res = solver.solve(callbacks=[NbIterationStopper(1)], time_limit=10)
    sol = res[-1][0]
    assert problem.satisfy(sol)
