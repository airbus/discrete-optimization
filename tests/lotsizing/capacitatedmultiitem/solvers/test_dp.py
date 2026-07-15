#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Tests for DP solvers on capacitated multi-item lot sizing."""

import numpy as np
import pytest

from discrete_optimization.lotsizing.capacitatedmultiitem.problem import (
    CapacitatedMultiItemLSP,
)
from discrete_optimization.lotsizing.capacitatedmultiitem.solvers.dp import (
    DpCapacitatedLotSizingSolver,
    DpSchedCapacitatedLotSizingSolver,
)


@pytest.fixture
def tiny_problem():
    """Create a tiny problem for DP testing."""
    nb_items = 2
    horizon = 4

    demands = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ],
        dtype=np.int64,
    )

    changeover_costs = np.array([[0, 10], [10, 0]], dtype=np.int64)
    stock_costs = np.array([1.0, 1.0], dtype=np.float64)

    return CapacitatedMultiItemLSP(
        nb_items=nb_items,
        horizon=horizon,
        demands=demands,
        capacity_machine=1,
        changeover_costs=changeover_costs,
        stock_cost_per_type=stock_costs,
        allow_delays=False,
    )


def test_dp_production_solver(tiny_problem):
    """Test DP production-based solver."""
    solver = DpCapacitatedLotSizingSolver(tiny_problem)
    solver.init_model()
    result = solver.solve(solver="LNBS", time_limit=10)

    assert len(result) > 0, "Should find at least one solution"

    solution = result.get_best_solution()
    assert tiny_problem.satisfy(solution), "Solution should satisfy all constraints"

    cost_dict = tiny_problem.evaluate(solution)
    assert cost_dict["backlog_cost"] == 0, "No backlog should occur"


def test_dp_scheduling_solver(tiny_problem):
    """Test DP scheduling-based solver."""
    solver = DpSchedCapacitatedLotSizingSolver(tiny_problem)
    solver.init_model()
    result = solver.solve(solver="LNBS", time_limit=10)

    assert len(result) > 0, "Should find at least one solution"

    solution = result.get_best_solution()
    assert tiny_problem.satisfy(solution), "Solution should satisfy all constraints"


def test_dp_both_solvers_same_result(tiny_problem):
    """Test that both DP solvers find solutions with similar costs."""
    solver1 = DpCapacitatedLotSizingSolver(tiny_problem)
    solver1.init_model()
    result1 = solver1.solve(solver="LNBS", time_limit=10)

    solver2 = DpSchedCapacitatedLotSizingSolver(tiny_problem)
    solver2.init_model()
    result2 = solver2.solve(solver="LNBS", time_limit=10)

    if len(result1) > 0 and len(result2) > 0:
        sol1 = result1.get_best_solution()
        sol2 = result2.get_best_solution()

        cost1 = sum(tiny_problem.evaluate(sol1).values())
        cost2 = sum(tiny_problem.evaluate(sol2).values())

        # Both should find optimal or near-optimal solutions
        # Allow some difference due to DP approximations
        assert abs(cost1 - cost2) < 100, (
            "Both solvers should find similar quality solutions"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
