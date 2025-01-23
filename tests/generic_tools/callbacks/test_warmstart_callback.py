#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Optional

import pytest

from discrete_optimization.generic_tools.callbacks.warm_start_callback import (
    WarmStartCallback,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.lexico_tools import LexicoSolver
from discrete_optimization.knapsack.parser import get_data_available, parse_file
from discrete_optimization.knapsack.problem import KnapsackProblem, MobjKnapsackModel
from discrete_optimization.knapsack.solvers.cpsat import CpSatKnapsackSolver


@pytest.mark.parametrize(
    "objectives",
    [("value", "weight", "heaviest_item")],
)
def test_knapsack_ortools_lexico_ws(objectives):
    model_file = [f for f in get_data_available() if "ks_60_0" in f][0]
    model: KnapsackProblem = parse_file(model_file, force_recompute_values=True)
    model: MobjKnapsackModel = MobjKnapsackModel.from_knapsack(model)
    subsolver = CpSatKnapsackSolver(model)
    solver = LexicoSolver(
        problem=model,
        subsolver=subsolver,
    )
    solver.init_model()
    mycb = WarmStartCallback(
        warm_start_last_solution=True, warm_start_best_solution=False
    )
    parameters_cp = ParametersCp.default()
    result_storage = solver.solve(
        time_limit=10,
        parameters_cp=parameters_cp,
        objectives=objectives,
        callbacks=[mycb],
    )
