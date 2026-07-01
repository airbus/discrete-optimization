#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Unit tests for uncapacitated single-item lot sizing problem and solution."""

from __future__ import annotations

from discrete_optimization.lotsizing.uncapacitatedsingleitem import (
    UncapacitatedSingleItemLSP,
    UncapacitatedSingleItemSolution,
    generate_random_instance,
    generate_wagner_whitin_example,
)


class TestUncapacitatedSingleItemProblem:
    """Test uncapacitated single-item problem construction and properties."""

    def test_basic_construction(self):
        """Test basic problem construction."""
        problem = UncapacitatedSingleItemLSP(
            demands=[10, 20, 15],
            setup_costs=[100.0, 100.0, 100.0],
            production_costs=[1.0, 1.0, 1.0],
            inventory_costs=[0.5, 0.5, 0.5],
        )

        assert problem.horizon == 3
        assert problem.nb_items == 1
        assert problem.items_list == [0]
        assert problem.get_total_demand(0) == 45

    def test_demand_access(self):
        """Test demand getters."""
        problem = UncapacitatedSingleItemLSP(
            demands=[10, 20, 15],
            setup_costs=[100.0] * 3,
            production_costs=[1.0] * 3,
            inventory_costs=[0.5] * 3,
        )

        assert problem.get_demand(0, 0) == 10
        assert problem.get_demand(0, 1) == 20
        assert problem.get_demand(0, 2) == 15

        cumul = problem.get_cumulative_demands(0)
        assert list(cumul) == [10, 30, 45]

    def test_cost_access(self):
        """Test cost getters."""
        problem = UncapacitatedSingleItemLSP(
            demands=[10, 20, 15],
            setup_costs=[50.0, 60.0, 70.0],
            production_costs=[1.0, 2.0, 3.0],
            inventory_costs=[0.5, 1.0, 1.5],
        )

        assert problem.get_setup_cost(0, 0) == 50.0
        assert problem.get_setup_cost(0, 1) == 60.0
        assert problem.get_production_cost_per_unit(0, 1) == 2.0
        assert problem.get_inventory_cost_per_unit(0, 2) == 1.5

    def test_wagner_whitin_example(self):
        """Test Wagner-Whitin classic example generation."""
        problem = generate_wagner_whitin_example()

        assert problem.horizon == 4
        demands = [problem.get_demand(0, t) for t in range(problem.horizon)]
        assert demands == [4, 8, 6, 7]

    def test_random_instance_generation(self):
        """Test random instance generation."""
        problem = generate_random_instance(horizon=10, seed=42)

        assert problem.horizon == 10
        assert problem.nb_items == 1

        # Check reproducibility
        problem2 = generate_random_instance(horizon=10, seed=42)
        for t in range(10):
            assert problem.get_demand(0, t) == problem2.get_demand(0, t)


class TestUncapacitatedSingleItemSolution:
    """Test uncapacitated single-item solution construction and methods."""

    def test_solution_construction(self):
        """Test basic solution construction."""
        problem = UncapacitatedSingleItemLSP(
            demands=[10, 20, 15],
            setup_costs=[100.0] * 3,
            production_costs=[1.0] * 3,
            inventory_costs=[0.5] * 3,
        )

        solution = UncapacitatedSingleItemSolution(
            problem=problem,
            production_periods=[0],
            production_quantities=[45],
        )

        assert len(solution.productions) == 1
        assert solution.productions[0].period == 0
        assert solution.productions[0].quantity == 45

    def test_inventory_computation(self):
        """Test automatic inventory and delivery computation."""
        problem = UncapacitatedSingleItemLSP(
            demands=[10, 20, 15],
            setup_costs=[100.0] * 3,
            production_costs=[1.0] * 3,
            inventory_costs=[0.5] * 3,
        )

        # Produce 30 in period 0, 15 in period 2
        solution = UncapacitatedSingleItemSolution(
            problem=problem,
            production_periods=[0, 2],
            production_quantities=[30, 15],
        )

        # Period 0: produce 30, demand 10, deliver 10, inventory 20
        assert solution.get_production_quantity(0, 0) == 30
        assert solution.get_delivery_quantity(0, 0) == 10
        assert solution.get_inventory_level(0, 0) == 20
        assert solution.get_backlog_quantity(0, 0) == 0

        # Period 1: produce 0, demand 20, deliver 20, inventory 0
        assert solution.get_production_quantity(0, 1) == 0
        assert solution.get_delivery_quantity(0, 1) == 20
        assert solution.get_inventory_level(0, 1) == 0
        assert solution.get_backlog_quantity(0, 1) == 0

        # Period 2: produce 15, demand 15, deliver 15, inventory 0
        assert solution.get_production_quantity(0, 2) == 15
        assert solution.get_delivery_quantity(0, 2) == 15
        assert solution.get_inventory_level(0, 2) == 0
        assert solution.get_backlog_quantity(0, 2) == 0

    def test_feasible_solution(self):
        """Test feasibility check on valid solution."""
        problem = UncapacitatedSingleItemLSP(
            demands=[10, 20, 15],
            setup_costs=[100.0] * 3,
            production_costs=[1.0] * 3,
            inventory_costs=[0.5] * 3,
        )

        solution = UncapacitatedSingleItemSolution(
            problem=problem,
            production_periods=[0, 2],
            production_quantities=[30, 15],
        )

        assert problem.satisfy(solution)
        assert solution.get_total_unmet_demand() == 0

    def test_infeasible_solution(self):
        """Test feasibility check on invalid solution."""
        problem = UncapacitatedSingleItemLSP(
            demands=[10, 20, 15],
            setup_costs=[100.0] * 3,
            production_costs=[1.0] * 3,
            inventory_costs=[0.5] * 3,
        )

        # Only produce 20, but total demand is 45
        solution = UncapacitatedSingleItemSolution(
            problem=problem,
            production_periods=[0],
            production_quantities=[20],
        )

        assert not problem.satisfy(solution)
        assert solution.get_total_unmet_demand() == 25

    def test_cost_computation(self):
        """Test cost computation."""
        problem = generate_wagner_whitin_example()

        # Produce 18 in period 0, 7 in period 3
        solution = UncapacitatedSingleItemSolution(
            problem=problem,
            production_periods=[0, 3],
            production_quantities=[18, 7],
        )

        # Setup cost: 2 setups × $15 = $30
        assert solution.compute_total_setup_cost() == 30.0

        # Production cost: 25 units × $1 = $25
        assert solution.compute_total_production_cost() == 25.0

        # Inventory cost
        assert solution.compute_total_inventory_cost() == 20.0

        # Backlog cost (should be 0)
        assert solution.compute_total_backlog_cost() == 0.0

        # Changeover cost (should be 0 for single item)
        assert solution.compute_total_changeover_cost() == 0.0

        # Total cost
        assert solution.compute_total_cost() == 75.0

    def test_evaluation(self):
        """Test problem evaluate() method."""
        problem = generate_wagner_whitin_example()

        solution = UncapacitatedSingleItemSolution(
            problem=problem,
            production_periods=[0, 2],
            production_quantities=[12, 13],
        )

        objectives = problem.evaluate(solution)

        assert "setup_cost" in objectives
        assert "production_cost" in objectives
        assert "inventory_cost" in objectives
        assert "backlog_cost" in objectives
        assert "changeover_cost" in objectives
        assert "unmet_demand" in objectives

        assert objectives["setup_cost"] == 30.0
        assert objectives["production_cost"] == 25.0
        assert objectives["inventory_cost"] == 15.0
        assert objectives["backlog_cost"] == 0.0
        assert objectives["changeover_cost"] == 0.0
        assert objectives["unmet_demand"] == 0.0


class TestEdgeCases:
    """Test edge cases."""

    def test_single_period(self):
        """Test single period problem."""
        problem = UncapacitatedSingleItemLSP(
            demands=[10],
            setup_costs=[50.0],
            production_costs=[1.0],
            inventory_costs=[0.5],
        )

        solution = UncapacitatedSingleItemSolution(
            problem=problem,
            production_periods=[0],
            production_quantities=[10],
        )

        assert problem.satisfy(solution)
        assert solution.compute_total_cost() == 60.0  # 50 + 10 + 0

    def test_zero_demands(self):
        """Test periods with zero demand."""
        problem = UncapacitatedSingleItemLSP(
            demands=[10, 0, 0, 5],
            setup_costs=[50.0] * 4,
            production_costs=[1.0] * 4,
            inventory_costs=[0.5] * 4,
        )

        solution = UncapacitatedSingleItemSolution(
            problem=problem,
            production_periods=[0],
            production_quantities=[15],
        )

        assert problem.satisfy(solution)

    def test_empty_solution(self):
        """Test empty solution (no production)."""
        problem = UncapacitatedSingleItemLSP(
            demands=[10, 20, 15],
            setup_costs=[100.0] * 3,
            production_costs=[1.0] * 3,
            inventory_costs=[0.5] * 3,
        )

        solution = UncapacitatedSingleItemSolution(
            problem=problem,
            production_periods=[],
            production_quantities=[],
        )

        assert not problem.satisfy(solution)
        assert solution.get_total_unmet_demand() == 45
