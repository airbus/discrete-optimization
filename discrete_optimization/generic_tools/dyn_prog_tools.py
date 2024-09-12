#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import didppy as dp

from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
)

solvers = {
    x.__name__: x
    for x in [
        dp.ForwardRecursion,
        dp.CABS,
        dp.CAASDy,
        dp.LNBS,
        dp.DFBB,
        dp.CBFS,
        dp.ACPS,
        dp.APPS,
        dp.DBDFS,
        dp.BreadthFirstSearch,
        dp.DDLNS,
        dp.WeightedAstar,
        dp.ExpressionBeamSearch,
    ]
}


class DidSolver(SolverDO):
    model: dp.Model
    hyperparameters = [CategoricalHyperparameter(name="solver", choices=solvers)]
