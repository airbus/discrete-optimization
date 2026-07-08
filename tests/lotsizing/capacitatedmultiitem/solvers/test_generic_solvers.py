#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Tests for generic solvers (DP, MILP, CP-SAT) on capacitated multi-item lot sizing."""

import numpy as np
import pytest

from discrete_optimization.lotsizing.capacitatedmultiitem.parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.lotsizing.capacitatedmultiitem.problem import (
    CapacitatedMultiItemLSP,
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
def tiny_problem():
    """Create a tiny problem for fast testing."""
    nb_items = 2
    horizon = 4

    demands = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.int64)
    changeover_costs = np.array([[0, 10], [10, 0]], dtype=np.int64)
    stock_costs = np.array([1.0, 1.0], dtype=np.float64)

    return CapacitatedMultiItemLSP(
        nb_items=nb_items,
        horizon=horizon,
        demands=demands,
        capacity_machine=2,
        changeover_costs=changeover_costs,
        stock_cost_per_type=stock_costs,
        allow_delays=False,
    )


@pytest.fixture
def pigment20a_instance():
    """Load pigment20a instance if available."""
    try:
        files = get_data_available()
        matching = [f for f in files if "pigment20a" in f.lower()]
        if matching:
            return parse_file(matching[0])
    except (FileNotFoundError, IndexError):
        pass
    return None


# ============================================================================
# Generic DP Solver Tests
# ============================================================================


def test_generic_dp_tiny(tiny_problem):
    """Test generic DP solver on tiny problem."""
    solver = GenericLotSizingDp(tiny_problem)
    solver.init_model()
    result = solver.solve(solver="LNBS", time_limit=5)

    assert len(result) > 0
    solution = result.get_best_solution()
    assert tiny_problem.satisfy(solution)


def test_generic_dp_with_dual_bounds(tiny_problem):
    """Test generic DP solver with dual bounds."""
    solver = GenericLotSizingDp(tiny_problem, add_additional_dual_bounds=True)
    solver.init_model()
    result = solver.solve(solver="LNBS", time_limit=5)

    assert len(result) > 0
    solution = result.get_best_solution()
    assert tiny_problem.satisfy(solution)


def test_generic_dp_pigment20a(pigment20a_instance):
    """Test generic DP solver on pigment20a."""
    if pigment20a_instance is None:
        pytest.skip("pigment20a instance not available")

    solver = GenericLotSizingDp(pigment20a_instance, add_additional_dual_bounds=True)
    solver.init_model()
    result = solver.solve(solver="LNBS", time_limit=10)

    assert len(result) > 0
    solution = result.get_best_solution()
    assert pigment20a_instance.satisfy(solution)


# ============================================================================
# Generic CP-SAT Solver Tests
# ============================================================================


def test_generic_cpsat_tiny(tiny_problem):
    """Test generic CP-SAT solver on tiny problem."""
    solver = GenericLotSizingCpsat(tiny_problem)
    solver.init_model()
    result = solver.solve(time_limit=5)

    assert len(result) > 0
    solution = result.get_best_solution()
    assert tiny_problem.satisfy(solution)


def test_generic_cpsat_pigment20a(pigment20a_instance):
    """Test generic CP-SAT solver on pigment20a."""
    if pigment20a_instance is None:
        pytest.skip("pigment20a instance not available")

    solver = GenericLotSizingCpsat(pigment20a_instance)
    solver.init_model()
    result = solver.solve(time_limit=10)

    assert len(result) > 0
    solution = result.get_best_solution()
    assert pigment20a_instance.satisfy(solution)


# ============================================================================
# Generic MILP Solver Tests
# ============================================================================


def test_generic_milp_tiny(tiny_problem):
    """Test generic MILP solver on tiny problem."""
    solver = MathOptGenericLotSizingMilp(tiny_problem)
    solver.init_model()
    result = solver.solve(time_limit=5)

    assert len(result) > 0
    solution = result.get_best_solution()
    assert tiny_problem.satisfy(solution)


def test_generic_milp_pigment20a(pigment20a_instance):
    """Test generic MILP solver on pigment20a."""
    if pigment20a_instance is None:
        pytest.skip("pigment20a instance not available")

    solver = MathOptGenericLotSizingMilp(pigment20a_instance)
    solver.init_model()
    result = solver.solve(time_limit=10)

    assert len(result) > 0
    solution = result.get_best_solution()
    assert pigment20a_instance.satisfy(solution)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
