#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Tests for CP-SAT solver on capacitated lot sizing with setup times."""

import numpy as np
import pytest

from discrete_optimization.lotsizing.capacitatedsetuptimes.problem import (
    CapacitatedSetupTimesLSP,
)
from discrete_optimization.lotsizing.capacitatedsetuptimes.solvers.cpsat import (
    CpSatSetupTimesSolver,
)


@pytest.fixture
def small_problem_with_setup():
    """Create a small problem with setup times."""
    nb_items = 2
    horizon = 4

    # Simple demands
    demands = np.array(
        [
            [2, 1, 1, 0],
            [1, 2, 0, 1],
        ],
        dtype=np.int64,
    )

    # Setup times: 2 units per setup
    setup_times = np.array(
        [
            [2.0, 2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0, 2.0],
        ],
        dtype=np.float64,
    )

    changeover_costs = np.array([[0, 10], [10, 0]], dtype=np.int64)
    stock_costs = np.array([1.0, 1.0], dtype=np.float64)

    return CapacitatedSetupTimesLSP(
        nb_items=nb_items,
        horizon=horizon,
        demands=demands,
        capacity_machine=10.0,  # Capacity: 10 units per period
        setup_times=setup_times,
        changeover_costs=changeover_costs,
        stock_cost_per_type=stock_costs,
        allow_delays=True,  # Allow backlog for feasibility
    )


@pytest.fixture
def varying_setup_times():
    """Create problem with varying setup times per item."""
    nb_items = 2
    horizon = 4

    demands = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.int64,
    )

    # Item 0 has longer setup, item 1 has shorter setup
    setup_times = np.array(
        [
            [3.0, 3.0, 3.0, 3.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    changeover_costs = np.array([[0, 5], [5, 0]], dtype=np.int64)
    stock_costs = np.array([1.0, 1.0], dtype=np.float64)

    return CapacitatedSetupTimesLSP(
        nb_items=nb_items,
        horizon=horizon,
        demands=demands,
        capacity_machine=10.0,
        setup_times=setup_times,
        changeover_costs=changeover_costs,
        stock_cost_per_type=stock_costs,
        allow_delays=True,
    )


def test_cpsat_with_setup_times(small_problem_with_setup):
    """Test CP-SAT solver with setup times."""
    solver = CpSatSetupTimesSolver(small_problem_with_setup)
    solver.init_model()
    result = solver.solve(time_limit=10)

    assert len(result) > 0, "Should find at least one solution"

    solution = result.get_best_solution()
    assert small_problem_with_setup.satisfy(solution), (
        "Solution should satisfy all constraints"
    )

    # Verify capacity constraints include setup times
    for t in range(small_problem_with_setup.horizon):
        total_time = solution.get_total_production_time_used(t)
        available = small_problem_with_setup.get_available_production_time(t)
        assert total_time <= available + 0.01, (
            f"Capacity exceeded at period {t}: {total_time} > {available}"
        )


def test_setup_times_consume_capacity(varying_setup_times):
    """Test that setup times correctly consume capacity."""
    solver = CpSatSetupTimesSolver(varying_setup_times)
    solver.init_model()
    result = solver.solve(time_limit=10)

    solution = result.get_best_solution()

    # Check that setup times are accounted for
    for t in range(varying_setup_times.horizon):
        for item in varying_setup_times.items_list:
            qty = solution.get_production_quantity(item, t)
            if qty > 0:
                # Verify that production + setup time <= capacity
                setup_time = varying_setup_times.get_setup_time(item, t)
                total_time = qty + setup_time

                # This individual item's time should be less than total capacity
                capacity = varying_setup_times.get_available_production_time(t)
                assert total_time <= capacity, (
                    f"Item {item} at period {t}: {total_time} > {capacity}"
                )


def test_varying_setup_times(varying_setup_times):
    """Test that solver handles varying setup times correctly."""
    solver = CpSatSetupTimesSolver(varying_setup_times)
    solver.init_model()
    result = solver.solve(time_limit=10)

    solution = result.get_best_solution()

    # Item 1 should be produced more often (shorter setup)
    # Item 0 should be produced less often (longer setup)

    item0_setups = sum(
        1
        for t in range(varying_setup_times.horizon)
        if solution.get_production_quantity(0, t) > 0
    )
    item1_setups = sum(
        1
        for t in range(varying_setup_times.horizon)
        if solution.get_production_quantity(1, t) > 0
    )

    # At least one item should be produced
    assert item0_setups + item1_setups > 0


def test_cost_components_with_setup_times(small_problem_with_setup):
    """Test that all cost components are computed correctly."""
    solver = CpSatSetupTimesSolver(small_problem_with_setup)
    solver.init_model()
    result = solver.solve(time_limit=10)

    solution = result.get_best_solution()
    cost_dict = small_problem_with_setup.evaluate(solution)

    # Check that we have all expected cost components
    assert "inventory_cost" in cost_dict
    assert "changeover_cost" in cost_dict
    assert "backlog_cost" in cost_dict

    # Setup costs should be 0 (we use WithoutSetupCostsProblem)
    assert cost_dict["setup_cost"] == 0.0

    # All costs should be non-negative
    for key, value in cost_dict.items():
        assert value >= 0, f"{key} should be non-negative"


def test_setup_time_solution_method(small_problem_with_setup):
    """Test that SetupTimesSolution correctly computes total production time."""
    solver = CpSatSetupTimesSolver(small_problem_with_setup)
    solver.init_model()
    result = solver.solve(time_limit=10)

    solution = result.get_best_solution()

    # Manually compute and verify
    for t in range(small_problem_with_setup.horizon):
        manual_total = 0.0
        for item in small_problem_with_setup.items_list:
            qty = solution.get_production_quantity(item, t)
            if qty > 0:
                # Production time
                manual_total += (
                    qty * small_problem_with_setup.get_production_time_per_unit(item, t)
                )
                # Setup time
                manual_total += small_problem_with_setup.get_setup_time(item, t)

        # Compare with solution method
        solution_total = solution.get_total_production_time_used(t)

        assert abs(manual_total - solution_total) < 0.01, (
            f"Production time mismatch at period {t}: {manual_total} vs {solution_total}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
