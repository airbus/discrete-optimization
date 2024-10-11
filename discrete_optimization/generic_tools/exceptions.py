#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


class SolveEarlyStop(Exception):
    """Exception used to stop some solvers

    See for instance discrete_optimization.gpdp.solver.ortools_solver.OrtoolsGpdpSolver.

    """

    ...
