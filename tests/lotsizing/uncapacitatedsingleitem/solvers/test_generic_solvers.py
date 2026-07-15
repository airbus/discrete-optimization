#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Tests for generic solvers (DP, MILP, CP-SAT) on uncapacitated single-item lot sizing."""

import pytest

from discrete_optimization.lotsizing.generic_solver.cpsat.generic_lotsizing_cpsat import (
    GenericLotSizingCpsat,
)
from discrete_optimization.lotsizing.generic_solver.dp.generic_dp_solver import (
    GenericLotSizingDp,
)
from discrete_optimization.lotsizing.generic_solver.milp.generic_lotsizing_milp import (
    MathOptGenericLotSizingMilp,
)
from discrete_optimization.lotsizing.uncapacitatedsingleitem.problem import (
    UncapacitatedSingleItemLSP,
)


@pytest.fixture
def tiny_uncapacitated_problem():
    """Create a tiny uncapacitated problem."""
    horizon = 6
    demands = [5, 7, 3, 6, 4, 8]
    setup_costs = [50.0] * horizon
    production_costs = [2.0] * horizon
    inventory_costs = [1.0] * horizon

    return UncapacitatedSingleItemLSP(
        demands=demands,
        setup_costs=setup_costs,
        production_costs=production_costs,
        inventory_costs=inventory_costs,
    )


@pytest.fixture
def small_uncapacitated_problem():
    """Create a small uncapacitated problem with varying costs."""
    horizon = 10
    demands = [3, 5, 2, 7, 4, 6, 3, 8, 2, 5]
    setup_costs = [60.0, 55.0, 50.0, 65.0, 60.0, 55.0, 50.0, 60.0, 55.0, 50.0]
    production_costs = [2.0, 2.2, 2.1, 2.3, 2.0, 2.1, 2.2, 2.0, 2.1, 2.0]
    inventory_costs = [1.0, 1.1, 1.0, 1.2, 1.0, 1.1, 1.0, 1.1, 1.0, 1.0]

    return UncapacitatedSingleItemLSP(
        demands=demands,
        setup_costs=setup_costs,
        production_costs=production_costs,
        inventory_costs=inventory_costs,
    )


# ============================================================================
# Generic DP Solver Tests
# ============================================================================


def test_generic_dp_tiny_uncapacitated(tiny_uncapacitated_problem):
    """Test generic DP solver on tiny uncapacitated problem."""
    solver = GenericLotSizingDp(tiny_uncapacitated_problem)
    solver.init_model()
    result = solver.solve(solver="LNBS", time_limit=5)

    assert len(result) > 0
    solution = result.get_best_solution()
    assert tiny_uncapacitated_problem.satisfy(solution)


def test_generic_dp_small_uncapacitated(small_uncapacitated_problem):
    """Test generic DP solver on small uncapacitated problem."""
    solver = GenericLotSizingDp(
        small_uncapacitated_problem, add_additional_dual_bounds=True
    )
    solver.init_model()
    result = solver.solve(solver="LNBS", time_limit=10)

    assert len(result) > 0
    solution = result.get_best_solution()
    assert small_uncapacitated_problem.satisfy(solution)


# ============================================================================
# Generic CP-SAT Solver Tests
# ============================================================================


def test_generic_cpsat_tiny_uncapacitated(tiny_uncapacitated_problem):
    """Test generic CP-SAT solver on tiny uncapacitated problem."""
    solver = GenericLotSizingCpsat(tiny_uncapacitated_problem)
    solver.init_model()
    result = solver.solve(time_limit=5)

    assert len(result) > 0
    solution = result.get_best_solution()
    assert tiny_uncapacitated_problem.satisfy(solution)


def test_generic_cpsat_small_uncapacitated(small_uncapacitated_problem):
    """Test generic CP-SAT solver on small uncapacitated problem."""
    solver = GenericLotSizingCpsat(small_uncapacitated_problem)
    solver.init_model()
    result = solver.solve(time_limit=10)

    assert len(result) > 0
    solution = result.get_best_solution()
    assert small_uncapacitated_problem.satisfy(solution)


# ============================================================================
# Generic MILP Solver Tests
# ============================================================================


def test_generic_milp_tiny_uncapacitated(tiny_uncapacitated_problem):
    """Test generic MILP solver on tiny uncapacitated problem."""
    solver = MathOptGenericLotSizingMilp(tiny_uncapacitated_problem)
    solver.init_model()
    result = solver.solve(time_limit=5)

    assert len(result) > 0
    solution = result.get_best_solution()
    assert tiny_uncapacitated_problem.satisfy(solution)


def test_generic_milp_small_uncapacitated(small_uncapacitated_problem):
    """Test generic MILP solver on small uncapacitated problem."""
    solver = MathOptGenericLotSizingMilp(small_uncapacitated_problem)
    solver.init_model()
    result = solver.solve(time_limit=10)

    assert len(result) > 0
    solution = result.get_best_solution()
    assert small_uncapacitated_problem.satisfy(solution)


# ============================================================================
# Solution Quality Tests
# ============================================================================


def test_uncapacitated_no_backlog(tiny_uncapacitated_problem):
    """Verify solutions have no backlog."""
    solver = GenericLotSizingCpsat(tiny_uncapacitated_problem)
    solver.init_model()
    result = solver.solve(time_limit=5)

    solution = result.get_best_solution()

    for t in range(tiny_uncapacitated_problem.horizon):
        backlog = solution.get_backlog_quantity(0, t)
        assert backlog == 0, f"Backlog should be 0 at period {t}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
