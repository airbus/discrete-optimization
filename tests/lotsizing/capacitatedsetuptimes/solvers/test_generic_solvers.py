#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Tests for generic solvers (DP, MILP, CP-SAT) on capacitated lot sizing with setup times."""

import numpy as np
import pytest

from discrete_optimization.lotsizing.capacitatedsetuptimes.problem import (
    CapacitatedSetupTimesLSP,
)
from discrete_optimization.lotsizing.generic_solver.cpsat.generic_lotsizing_cpsat import (
    GenericLotSizingCpsat,
)
from discrete_optimization.lotsizing.generic_solver.dp.generic_dp_solver import (
    GenericLotSizingDp,
)
from discrete_optimization.lotsizing.generic_solver.milp.generic_lotsizing_milp import (
    MathOptGenericLotSizingMilp,
)


@pytest.fixture
def tiny_setup_times_problem():
    """Create a tiny problem with setup times."""
    nb_items = 2
    horizon = 4

    # Demands
    demands = np.array([[2, 0, 1, 0], [0, 1, 0, 2]], dtype=np.int64)

    # Setup times (consume capacity)
    setup_times = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])

    # Inventory costs
    stock_costs = np.array([1.0, 1.0])

    return CapacitatedSetupTimesLSP(
        nb_items=nb_items,
        horizon=horizon,
        demands=demands,
        capacity_machine=5.0,  # Capacity per period
        setup_times=setup_times,
        stock_cost_per_type=stock_costs,
        allow_delays=False,
    )


@pytest.fixture
def small_setup_times_problem():
    """Create a small problem with setup times and more complexity."""
    nb_items = 3
    horizon = 6

    demands = np.array(
        [[1, 0, 2, 0, 1, 0], [0, 1, 0, 2, 0, 1], [1, 0, 1, 0, 1, 1]], dtype=np.int64
    )

    setup_times = np.ones((nb_items, horizon)) * 2.0
    stock_costs = np.array([1.5, 1.0, 2.0])

    return CapacitatedSetupTimesLSP(
        nb_items=nb_items,
        horizon=horizon,
        demands=demands,
        capacity_machine=10.0,  # Capacity per period
        setup_times=setup_times,
        stock_cost_per_type=stock_costs,
        allow_delays=False,
    )


# ============================================================================
# Generic DP Solver Tests
# ============================================================================


def test_generic_dp_tiny_setup_times(tiny_setup_times_problem):
    """Test generic DP solver on tiny setup times problem."""
    solver = GenericLotSizingDp(tiny_setup_times_problem)
    solver.init_model()
    result = solver.solve(solver="LNBS", time_limit=5)

    assert len(result) > 0
    solution = result.get_best_solution()
    assert tiny_setup_times_problem.satisfy(solution)


def test_generic_dp_small_setup_times(small_setup_times_problem):
    """Test generic DP solver on small setup times problem."""
    solver = GenericLotSizingDp(
        small_setup_times_problem, add_additional_dual_bounds=True
    )
    solver.init_model()
    result = solver.solve(solver="LNBS", time_limit=10)

    assert len(result) > 0
    solution = result.get_best_solution()
    assert small_setup_times_problem.satisfy(solution)


# ============================================================================
# Generic CP-SAT Solver Tests
# ============================================================================


def test_generic_cpsat_tiny_setup_times(tiny_setup_times_problem):
    """Test generic CP-SAT solver on tiny setup times problem."""
    solver = GenericLotSizingCpsat(tiny_setup_times_problem)
    solver.init_model()
    result = solver.solve(time_limit=5)

    assert len(result) > 0
    solution = result.get_best_solution()
    assert tiny_setup_times_problem.satisfy(solution)


def test_generic_cpsat_small_setup_times(small_setup_times_problem):
    """Test generic CP-SAT solver on small setup times problem."""
    solver = GenericLotSizingCpsat(small_setup_times_problem)
    solver.init_model()
    result = solver.solve(time_limit=10)

    assert len(result) > 0
    solution = result.get_best_solution()
    assert small_setup_times_problem.satisfy(solution)


# ============================================================================
# Generic MILP Solver Tests
# ============================================================================


def test_generic_milp_tiny_setup_times(tiny_setup_times_problem):
    """Test generic MILP solver on tiny setup times problem."""
    solver = MathOptGenericLotSizingMilp(tiny_setup_times_problem)
    solver.init_model()
    result = solver.solve(time_limit=5)

    assert len(result) > 0
    solution = result.get_best_solution()
    assert tiny_setup_times_problem.satisfy(solution)


def test_generic_milp_small_setup_times(small_setup_times_problem):
    """Test generic MILP solver on small setup times problem."""
    solver = MathOptGenericLotSizingMilp(small_setup_times_problem)
    solver.init_model()
    result = solver.solve(time_limit=10)

    assert len(result) > 0
    solution = result.get_best_solution()
    assert small_setup_times_problem.satisfy(solution)


# ============================================================================
# Solution Quality Tests
# ============================================================================


def test_setup_times_capacity_constraints(tiny_setup_times_problem):
    """Verify solutions respect capacity including setup times."""
    solver = GenericLotSizingCpsat(tiny_setup_times_problem)
    solver.init_model()
    result = solver.solve(time_limit=5)

    solution = result.get_best_solution()

    # Check capacity including setup times
    for t in range(tiny_setup_times_problem.horizon):
        time_used = solution.get_total_production_time_used(t)
        capacity = tiny_setup_times_problem.get_available_production_time(t)
        assert time_used <= capacity, (
            f"Time used {time_used} exceeds capacity {capacity} at period {t}"
        )


def test_setup_times_dp_vs_cpsat(tiny_setup_times_problem):
    """Test DP and CP-SAT find similar quality solutions."""
    dp_solver = GenericLotSizingDp(tiny_setup_times_problem)
    dp_solver.init_model()
    dp_result = dp_solver.solve(solver="LNBS", time_limit=5)

    cpsat_solver = GenericLotSizingCpsat(tiny_setup_times_problem)
    cpsat_solver.init_model()
    cpsat_result = cpsat_solver.solve(time_limit=5)

    if len(dp_result) > 0 and len(cpsat_result) > 0:
        dp_cost = sum(
            tiny_setup_times_problem.evaluate(dp_result.get_best_solution()).values()
        )
        cpsat_cost = sum(
            tiny_setup_times_problem.evaluate(cpsat_result.get_best_solution()).values()
        )

        # Allow 30% difference
        assert abs(dp_cost - cpsat_cost) <= max(dp_cost, cpsat_cost) * 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
