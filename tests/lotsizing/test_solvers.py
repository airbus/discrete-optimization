#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest

from discrete_optimization.lotsizing.problem import (
    LotSizingProblem,
    LotSizingSolution,
)
from discrete_optimization.lotsizing.solvers import (
    CpSatLotSizingSolver,
    DpLotSizingSolver,
    MathOptLotSizingSolver,
)


@pytest.fixture
def tiny_problem():
    """Create a tiny lot sizing problem for testing."""
    return LotSizingProblem(
        nb_items_type=2,
        capacity_machine=1,
        changeover_costs=[[0, 1], [1, 0]],
        demands=[[0, 1, 0, 0, 1], [1, 0, 0, 0, 1]],
        stock_capacity=10,
        stock_cost_per_type_per_time_per_unit=[1, 1],
        delay_cost_per_type_per_time_per_unit=[1, 1],
        allow_delays=False,
    )


def test_cpsat_solver(tiny_problem):
    """Test CP-SAT solver on tiny problem."""
    solver = CpSatLotSizingSolver(tiny_problem)
    res = solver.solve(time_limit=5)

    assert len(res) > 0
    sol: LotSizingSolution = res[-1][0]

    # Check solution is valid
    assert tiny_problem.satisfy(sol)

    # Check we have productions
    assert len(sol.productions) > 0

    # Check cost is reasonable
    cost = tiny_problem.evaluate(sol)
    assert "stock" in cost
    assert "changeover" in cost
    assert "delays" in cost


def test_mathopt_solver(tiny_problem):
    """Test MathOpt MILP solver on tiny problem."""
    solver = MathOptLotSizingSolver(tiny_problem)
    res = solver.solve(time_limit=5)

    assert len(res) > 0
    sol: LotSizingSolution = res[-1][0]

    # Check solution is valid
    assert tiny_problem.satisfy(sol)

    # Check we have productions
    assert len(sol.productions) > 0


def test_dp_solver(tiny_problem):
    """Test DP solver on tiny problem."""
    solver = DpLotSizingSolver(tiny_problem)
    res = solver.solve(time_limit=5, solver="LNBS")

    assert len(res) > 0
    sol: LotSizingSolution = res[-1][0]

    # Check solution is valid
    assert tiny_problem.satisfy(sol)

    # Check we have productions
    assert len(sol.productions) > 0


def test_solvers_agree(tiny_problem):
    """Test that different solvers find similar quality solutions."""
    solvers = [
        CpSatLotSizingSolver(tiny_problem),
        MathOptLotSizingSolver(tiny_problem),
    ]

    costs = []
    for solver in solvers:
        res = solver.solve(time_limit=10)
        sol = res[-1][0]
        cost = sum(tiny_problem.evaluate(sol).values())
        costs.append(cost)

    # Both solvers should find the optimal or near-optimal solution
    assert all(
        tiny_problem.satisfy(res[-1][0])
        for res in [s.solve(time_limit=10) for s in solvers]
    )
    # Costs should be the same or very close
    assert max(costs) - min(costs) <= 1  # Allow small differences
