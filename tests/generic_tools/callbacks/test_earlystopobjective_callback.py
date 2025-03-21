#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest
from ortools.math_opt.python import mathopt

from discrete_optimization.coloring.parser import (
    get_data_available as coloring_get_data_available,
)
from discrete_optimization.coloring.parser import parse_file as coloring_parse_file
from discrete_optimization.coloring.solvers.lp import (
    GurobiColoringSolver,
    MathOptColoringSolver,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    ObjectiveGapStopper,
)
from discrete_optimization.generic_tools.callbacks.stats_retrievers import (
    StatsWithBoundsCallback,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.knapsack.parser import (
    get_data_available as knapsack_get_data_available,
)
from discrete_optimization.knapsack.parser import parse_file as knapsack_parse_file
from discrete_optimization.knapsack.problem import KnapsackProblem, MobjKnapsackModel
from discrete_optimization.knapsack.solvers.cpsat import CpSatKnapsackSolver

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True


def test_knapsack_ortools_cpsat_gap_callback():
    model_file = [f for f in knapsack_get_data_available() if "ks_300_0" in f][0]
    model: KnapsackProblem = knapsack_parse_file(
        model_file, force_recompute_values=True
    )
    model: MobjKnapsackModel = MobjKnapsackModel.from_knapsack(model)

    # w/o stop
    solver = CpSatKnapsackSolver(model)
    solver.init_model()
    result_storage = solver.solve(
        time_limit=10,
        ortools_cpsat_solver_kwargs={"log_search_progress": True},
    )
    nb_solutions_wo_gap_stop = len(result_storage)

    # with gap stop
    solver = CpSatKnapsackSolver(model)
    solver.init_model()
    objective_gap_rel = 0.1
    objective_gap_abs = 10
    mycb = ObjectiveGapStopper(
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
    assert len(result_storage) < nb_solutions_wo_gap_stop


@pytest.mark.parametrize(
    "solver_type", [mathopt.SolverType.GSCIP, mathopt.SolverType.CP_SAT]
)
def test_coloring_mathopt_gap_callback(solver_type):
    file = [f for f in coloring_get_data_available() if "gc_70_1" in f][0]
    color_problem = coloring_parse_file(file)

    solver = MathOptColoringSolver(
        color_problem,
    )
    sol = color_problem.get_dummy_solution()
    solver.init_model(greedy_start=False)
    solver.set_warm_start(sol)

    # w/o early stopping
    stats_cb = StatsWithBoundsCallback()

    result_store = solver.solve(mathopt_solver_type=solver_type, callbacks=[stats_cb])
    nb_solutions_wo_gap_stop = len(result_store)
    print(len(result_store), stats_cb.stats)
    assert len(stats_cb.stats) == len(result_store) + 1

    # with stopping based on gap
    objective_gap_abs = 70
    objective_gap_rel = None
    stopper_cb = ObjectiveGapStopper(
        objective_gap_rel=objective_gap_rel, objective_gap_abs=objective_gap_abs
    )
    stats_cb = StatsWithBoundsCallback()
    result_store = solver.solve(
        mathopt_solver_type=solver_type, callbacks=[stopper_cb, stats_cb]
    )

    print(len(result_store), stats_cb.stats)
    assert len(stats_cb.stats) == len(result_store) + 1
    assert all(
        "obj" in stats_item and "bound" in stats_item for stats_item in stats_cb.stats
    )

    if solver_type != mathopt.SolverType.CP_SAT:  # cp_sat does not provide bounds
        assert len(result_store) < nb_solutions_wo_gap_stop

    # with stopping based on rel gap
    objective_gap_abs = None
    objective_gap_rel = 2
    stopper_cb = ObjectiveGapStopper(
        objective_gap_rel=objective_gap_rel, objective_gap_abs=objective_gap_abs
    )
    stats_cb = StatsWithBoundsCallback()
    result_store = solver.solve(
        mathopt_solver_type=solver_type, callbacks=[stopper_cb, stats_cb]
    )

    print(len(result_store), stats_cb.stats)
    assert len(stats_cb.stats) == len(result_store) + 1
    assert all(
        "obj" in stats_item and "bound" in stats_item for stats_item in stats_cb.stats
    )

    if solver_type not in (
        mathopt.SolverType.CP_SAT,
        mathopt.SolverType.GSCIP,
    ):  # GSCIP go on 1 step before actually stopping...
        assert len(result_store) < nb_solutions_wo_gap_stop


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_coloring_gurobi_gap_callback():
    file = [f for f in coloring_get_data_available() if "gc_20_1" in f][0]
    color_problem = coloring_parse_file(file)

    # w/o early stopping
    solver = GurobiColoringSolver(
        color_problem,
    )
    sol = color_problem.get_dummy_solution()
    solver.init_model(greedy_start=False)
    solver.set_warm_start(sol)
    stats_cb = StatsWithBoundsCallback()
    result_store = solver.solve(callbacks=[stats_cb])
    nb_solutions_wo_gap_stop = len(result_store)
    print(len(result_store), stats_cb.stats)
    assert len(stats_cb.stats) == len(result_store) + 1

    # with stopping based on gap
    objective_gap_abs = 3e100
    objective_gap_rel = None
    solver = GurobiColoringSolver(
        color_problem,
    )
    sol = color_problem.get_dummy_solution()
    solver.init_model(greedy_start=False)
    solver.set_warm_start(sol)
    stopper_cb = ObjectiveGapStopper(
        objective_gap_rel=objective_gap_rel, objective_gap_abs=objective_gap_abs
    )
    stats_cb = StatsWithBoundsCallback()
    result_store = solver.solve(callbacks=[stopper_cb, stats_cb])

    print(stats_cb.stats)
    assert len(stats_cb.stats) == len(result_store) + 1
    assert all(
        "obj" in stats_item and "bound" in stats_item for stats_item in stats_cb.stats
    )
    assert len(result_store) < nb_solutions_wo_gap_stop

    # with stopping based on rel gap
    objective_gap_abs = None
    objective_gap_rel = 3
    solver = GurobiColoringSolver(
        color_problem,
    )
    sol = color_problem.get_dummy_solution()
    solver.init_model(greedy_start=False)
    solver.set_warm_start(sol)
    stopper_cb = ObjectiveGapStopper(
        objective_gap_rel=objective_gap_rel, objective_gap_abs=objective_gap_abs
    )
    stats_cb = StatsWithBoundsCallback()
    result_store = solver.solve(callbacks=[stopper_cb, stats_cb])

    print(stats_cb.stats)
    assert len(stats_cb.stats) == len(result_store) + 1
    assert all(
        "obj" in stats_item and "bound" in stats_item for stats_item in stats_cb.stats
    )
    print(stats_cb.stats)
    assert len(result_store) < nb_solutions_wo_gap_stop
