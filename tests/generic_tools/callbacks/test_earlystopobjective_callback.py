#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    ObjectiveGapCpSatSolver,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.knapsack.parser import get_data_available, parse_file
from discrete_optimization.knapsack.problem import KnapsackProblem, MobjKnapsackModel
from discrete_optimization.knapsack.solvers.cpsat import CpSatKnapsackSolver


def test_knapsack_ortools_objective_callback():
    model_file = [f for f in get_data_available() if "ks_300_0" in f][0]
    model: KnapsackProblem = parse_file(model_file, force_recompute_values=True)
    model: MobjKnapsackModel = MobjKnapsackModel.from_knapsack(model)
    solver = CpSatKnapsackSolver(model)
    solver.init_model()
    objective_gap_rel = 0.1
    objective_gap_abs = 10
    mycb = ObjectiveGapCpSatSolver(
        objective_gap_rel=objective_gap_rel, objective_gap_abs=objective_gap_abs
    )
    parameters_cp = ParametersCp.default()
    result_storage = solver.solve(
        time_limit=10,
        parameters_cp=parameters_cp,
        callbacks=[mycb],
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
    )
    assert (
        abs(solver.clb.ObjectiveValue() - solver.clb.BestObjectiveBound())
        <= objective_gap_abs
        or abs(solver.clb.ObjectiveValue() - solver.clb.BestObjectiveBound())
        / abs(solver.clb.BestObjectiveBound())
        <= objective_gap_rel
    )
