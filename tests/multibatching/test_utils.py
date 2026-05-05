#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Tests for multibatching problem generation utilities."""

from discrete_optimization.multibatching.utils import generate_multibatching_problem


class TestGenerateMultibatchingProblem:
    """Test the random problem generator."""

    def test_basic_generation(self):
        """Test that a basic problem can be generated."""
        problem = generate_multibatching_problem(
            num_locations=3,
            num_transport_types=2,
            num_products=2,
            seed=42,
        )

        assert problem.nb_locations == 3
        assert problem.nb_transport_types == 2
        assert problem.nb_products == 2
        assert problem.nb_transport_links > 0

    def test_seed_reproducibility(self):
        """Test that using the same seed produces the same problem."""
        problem1 = generate_multibatching_problem(
            num_locations=4,
            num_transport_types=2,
            num_products=2,
            seed=123,
        )

        problem2 = generate_multibatching_problem(
            num_locations=4,
            num_transport_types=2,
            num_products=2,
            seed=123,
        )

        # Check that key characteristics are the same
        assert problem1.nb_transport_links == problem2.nb_transport_links
        assert len(problem1.products) == len(problem2.products)
        assert len(problem1.locations) == len(problem2.locations)

        # Check that supply/demand are the same
        for i, (loc1, loc2) in enumerate(zip(problem1.locations, problem2.locations)):
            for prod1, prod2 in zip(problem1.products, problem2.products):
                assert loc1.net_supply.get(prod1, 0) == loc2.net_supply.get(prod2, 0)

    def test_supply_demand_balance(self):
        """Test that total supply equals total demand for each product."""
        problem = generate_multibatching_problem(
            num_locations=5,
            num_transport_types=2,
            num_products=3,
            seed=42,
        )

        # For each product, sum of net_supply across all locations should be 0
        for product in problem.products:
            total_net_supply = sum(
                loc.net_supply.get(product, 0) for loc in problem.locations
            )
            assert abs(total_net_supply) < 1e-6, (
                f"Product {product.id} has imbalanced supply/demand: {total_net_supply}"
            )

    def test_transport_links_structure(self):
        """Test that transport links are correctly structured."""
        problem = generate_multibatching_problem(
            num_locations=3,
            num_transport_types=2,
            num_products=2,
            max_links_per_pair=1,
            seed=42,
        )

        # Check that transport links connect valid locations
        for link in problem.transport_links:
            assert link.location_l1 in problem.locations
            assert link.location_l2 in problem.locations
            assert link.location_l1 != link.location_l2  # No self-loops
            assert link.transport_type in problem.transport_types
            assert link.distance > 0

    def test_product_transport_compatibility(self):
        """Test that products have valid transport types."""
        problem = generate_multibatching_problem(
            num_locations=4,
            num_transport_types=3,
            num_products=2,
            seed=42,
        )

        # Each product should have at least one valid transport
        for product in problem.products:
            assert len(product.valid_transports) > 0
            # All valid transports should be in the problem's transport types
            for transport in product.valid_transports:
                assert transport in problem.transport_types

    def test_custom_parameters(self):
        """Test generation with custom parameter ranges."""
        problem = generate_multibatching_problem(
            num_locations=3,
            num_transport_types=2,
            num_products=2,
            min_capacity=50,
            max_capacity=100,
            min_product_size=10,
            max_product_size=20,
            seed=42,
        )

        # Check that transport types have capacity in the right range
        for tt in problem.transport_types:
            assert 50 <= tt.capacity <= 100

        # Check that products have size in the right range
        for prod in problem.products:
            assert 10 <= prod.size <= 20

    def test_feasibility_check(self):
        """Test that the generated problem can check feasibility."""
        problem = generate_multibatching_problem(
            num_locations=4,
            num_transport_types=2,
            num_products=2,
            seed=42,
        )

        # The problem should be able to check feasibility per product
        is_feasible = problem.check_feasibility_per_product(verbose=False)
        # This should not raise an exception
        assert isinstance(is_feasible, bool)
