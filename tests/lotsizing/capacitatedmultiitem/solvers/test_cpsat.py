#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Tests for CP-SAT solver on capacitated multi-item lot sizing."""

import numpy as np
import pytest

from discrete_optimization.lotsizing.capacitatedmultiitem.problem import (
    CapacitatedMultiItemLSP,
)
from discrete_optimization.lotsizing.capacitatedmultiitem.solvers.cpsat import (
    ChangeoverModel,
    CpSatCapacitatedLotSizingSolver,
)


@pytest.fixture
def small_problem():
    """Create a small feasible problem instance."""
    nb_items = 2
    horizon = 4

    # Small demands - total demand = 4, capacity = 4*2 = 8 (feasible)
    demands = np.array(
        [
            [1, 0, 1, 0],  # Item 0: needs 2 total
            [0, 1, 0, 1],  # Item 1: needs 2 total
        ],
        dtype=np.int64,
    )

    # Changeover costs
    changeover_costs = np.array(
        [
            [0, 10],
            [10, 0],
        ],
        dtype=np.int64,
    )

    # Stock costs
    stock_costs = np.array([1.0, 1.0], dtype=np.float64)

    return CapacitatedMultiItemLSP(
        nb_items=nb_items,
        horizon=horizon,
        demands=demands,
        capacity_machine=2,  # 2 units per period
        changeover_costs=changeover_costs,
        stock_cost_per_type=stock_costs,
        allow_delays=False,  # No backlog allowed
    )


@pytest.fixture
def problem_with_backlog():
    """Create a problem that may need backlog."""
    nb_items = 2
    horizon = 4

    # Demands that are tight on capacity
    demands = np.array(
        [
            [2, 1, 2, 1],
            [1, 2, 1, 2],
        ],
        dtype=np.int64,
    )

    changeover_costs = np.array([[0, 15], [15, 0]], dtype=np.int64)
    stock_costs = np.array([1.0, 1.0], dtype=np.float64)

    return CapacitatedMultiItemLSP(
        nb_items=nb_items,
        horizon=horizon,
        demands=demands,
        capacity_machine=3,
        changeover_costs=changeover_costs,
        stock_cost_per_type=stock_costs,
        allow_delays=True,
        delay_cost_per_type=np.array([100.0, 100.0]),
    )


def test_cpsat_state_based_small_problem(small_problem):
    """Test CP-SAT solver with state-based changeover model on small problem."""
    solver = CpSatCapacitatedLotSizingSolver(small_problem)
    solver.init_model(changeover_model=ChangeoverModel.STATE_BASED)
    result = solver.solve(time_limit=10)

    assert len(result) > 0, "Should find at least one solution"

    solution = result.get_best_solution()

    # Verify solution is valid
    assert small_problem.satisfy(solution), "Solution should satisfy all constraints"

    # Verify cost structure
    cost_dict = small_problem.evaluate(solution)
    assert cost_dict["inventory_cost"] >= 0
    assert cost_dict["changeover_cost"] >= 0
    assert cost_dict["backlog_cost"] == 0, "No backlog allowed"

    # Verify capacity constraints
    for t in range(small_problem.horizon):
        capacity_used = solution.get_total_production_time_used(t)
        capacity_available = small_problem.get_available_production_time(t)
        assert capacity_used <= capacity_available, f"Capacity exceeded at period {t}"


def test_cpsat_shortest_path_small_problem(small_problem):
    """Test CP-SAT solver with shortest path changeover model."""
    solver = CpSatCapacitatedLotSizingSolver(small_problem)
    solver.init_model(changeover_model=ChangeoverModel.SHORTEST_PATH_BASED)
    result = solver.solve(time_limit=10)

    assert len(result) > 0, "Should find at least one solution"

    solution = result.get_best_solution()
    assert small_problem.satisfy(solution), "Solution should satisfy all constraints"


def test_cpsat_with_backlog(problem_with_backlog):
    """Test CP-SAT solver on problem allowing backlog."""
    solver = CpSatCapacitatedLotSizingSolver(problem_with_backlog)
    solver.init_model(changeover_model=ChangeoverModel.STATE_BASED)
    result = solver.solve(time_limit=10)

    assert len(result) > 0, "Should find at least one solution"

    solution = result.get_best_solution()
    assert problem_with_backlog.satisfy(solution), (
        "Solution should satisfy all constraints"
    )

    cost_dict = problem_with_backlog.evaluate(solution)

    # Backlog is allowed, but we want to minimize it
    # Check that total cost is reasonable
    total_cost = sum(cost_dict.values())
    assert total_cost >= 0


def test_cpsat_production_quantities(small_problem):
    """Test that production quantities respect capacity."""
    solver = CpSatCapacitatedLotSizingSolver(small_problem)
    solver.init_model()
    result = solver.solve(time_limit=10)

    solution = result.get_best_solution()

    # Check each period
    for t in range(small_problem.horizon):
        total_production = sum(
            solution.get_production_quantity(item, t)
            for item in small_problem.items_list
        )
        assert total_production <= small_problem.capacity_machine


def test_cpsat_no_backlog_constraint(small_problem):
    """Test that no backlog constraint is enforced."""
    solver = CpSatCapacitatedLotSizingSolver(small_problem)
    solver.init_model()
    result = solver.solve(time_limit=10)

    solution = result.get_best_solution()

    # Verify no backlog
    for item in small_problem.items_list:
        for t in range(small_problem.horizon):
            backlog = solution.get_backlog_quantity(item, t)
            assert backlog == 0, f"Backlog should be 0 for item {item} at period {t}"


def test_cost_evolution(small_problem):
    """Test that cost evolution method works correctly."""
    solver = CpSatCapacitatedLotSizingSolver(small_problem)
    solver.init_model()
    result = solver.solve(time_limit=10)

    solution = result.get_best_solution()

    # Get cost evolution
    cost_evolution = solution.get_cost_evolution()

    # Check structure
    assert "inventory" in cost_evolution
    assert "backlog" in cost_evolution
    assert "changeover" in cost_evolution
    assert "total" in cost_evolution

    # Check that costs are cumulative (monotonically increasing)
    for key in cost_evolution:
        costs = cost_evolution[key]
        assert len(costs) == small_problem.horizon
        for i in range(1, len(costs)):
            assert costs[i] >= costs[i - 1], f"{key} cost should be cumulative"

    # Check that total matches evaluate
    cost_dict = small_problem.evaluate(solution)
    final_total = cost_evolution["total"][-1]

    # Allow small numerical differences
    assert abs(final_total - sum(cost_dict.values())) < 1.0, (
        "Cost evolution total should match evaluate()"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
