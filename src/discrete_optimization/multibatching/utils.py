import random
from collections import defaultdict

from discrete_optimization.multibatching.problem import (
    Location,
    MultibatchingProblem,
    Product,
    TransportLink,
    TransportType,
)


def generate_multibatching_problem(
    num_locations: int = 6,
    num_transport_types: int = 2,
    num_products: int = 2,
    min_dist: int = 1,
    max_dist: int = 10,
    min_capacity: int = 10,
    max_capacity: int = 20,
    min_product_size: int = 1,
    max_product_size: int = 5,
    min_product_value: int = 100,
    max_product_value: int = 1000,
    min_tt_cost: int = 10,
    max_tt_cost: int = 100,
    min_tt_speed: float = 0.1,
    max_tt_speed: float = 1.0,
    min_tt_emissions: int = 10,
    max_tt_emissions: int = 100,
    max_links_per_pair: int = 2,
    demand_sparsity: float = 0.5,
    max_demand_abs: int = 50,
    seed: int = None,
) -> MultibatchingProblem:
    """Generate a random MultibatchingProblem instance.

    Creates a multibatching logistics problem with random transport networks, products,
    locations, and supply/demand. The problem models the optimization of product transport
    across multiple locations using different transport types (e.g., trucks, trains),
    considering costs, emissions, and capacity constraints.

    The generator ensures:
    - Supply/demand balance: Total supply equals total demand for each product
    - Product-transport compatibility: Each product can only use valid transport types
    - Network connectivity: Transport links connect locations based on available transport types

    Args:
        num_locations: Number of distinct locations in the network.
        num_transport_types: Number of distinct transport resource types (e.g., truck, train).
        num_products: Number of distinct products to transport.
        min_dist: Minimum distance for transport links.
        max_dist: Maximum distance for transport links.
        min_capacity: Minimum capacity for transport types.
        max_capacity: Maximum capacity for transport types.
        min_product_size: Minimum size/volume for products.
        max_product_size: Maximum size/volume for products.
        min_product_value: Minimum value for products.
        max_product_value: Maximum value for products.
        min_tt_cost: Minimum cost per unit distance for transport types.
        max_tt_cost: Maximum cost per unit distance for transport types.
        min_tt_speed: Minimum speed for transport types.
        max_tt_speed: Maximum speed for transport types.
        min_tt_emissions: Minimum emissions per unit distance for transport types.
        max_tt_emissions: Maximum emissions per unit distance for transport types.
        max_links_per_pair: Maximum number of transport links (with different transport types)
            between any pair of locations. Setting this to 1 creates a simple network where
            each location pair has at most one transport option.
        demand_sparsity: Proportion of (location, product) pairs that will have non-zero
            supply/demand (0.0 = no demand anywhere, 1.0 = all pairs have demand).
        max_demand_abs: Maximum absolute value for supply (+) or demand (-) at any location
            for any product.
        seed: Random seed for reproducibility. If None, uses system time.

    Returns:
        A randomly generated MultibatchingProblem instance with balanced supply/demand.

    Example:
        >>> from discrete_optimization.multibatching.utils import generate_multibatching_problem
        >>> problem = generate_multibatching_problem(
        ...     num_locations=5,
        ...     num_transport_types=2,
        ...     num_products=3,
        ...     seed=42
        ... )
        >>> print(f"Problem has {problem.nb_locations} locations")
        >>> print(f"Problem has {problem.nb_transport_links} transport links")
    """
    if seed is not None:
        random.seed(seed)

    # 1. Generate TransportType instances
    transport_types_list = []
    for i in range(num_transport_types):
        tt = TransportType(
            id=i + 1,
            cost=random.randint(min_tt_cost, max_tt_cost),
            speed=round(random.uniform(min_tt_speed, max_tt_speed), 1),
            emissions=random.randint(min_tt_emissions, max_tt_emissions),
            capacity=random.randint(min_capacity, max_capacity),
            name=f"transport_{i}",
        )
        transport_types_list.append(tt)

    # 2. Generate Product instances
    products_list = []
    for i in range(num_products):
        # Each product can be transported by a random subset of transport types
        num_valid_tts = random.randint(1, len(transport_types_list))
        valid_transports = frozenset(random.sample(transport_types_list, num_valid_tts))
        prod = Product(
            id=i + 1,
            size=random.randint(min_product_size, max_product_size),
            value=random.randint(min_product_value, max_product_value),
            valid_transports=valid_transports,
            name=f"p{i + 1}",
        )
        products_list.append(prod)

    # 3. Generate Location instances with balanced net_supply
    locations_list = []
    product_global_demands = defaultdict(
        int
    )  # To keep track of total supply/demand for each product

    # First, generate demands that sum to zero for each product
    # Create potential supply/demand points
    temp_demands = defaultdict(lambda: defaultdict(int))  # {loc_id: {prod: qty}}
    for i in range(num_locations):
        loc_id = f"l{i + 1}"
        for prod in products_list:
            if random.random() < demand_sparsity:  # Introduce sparsity
                demand_val = random.randint(-max_demand_abs, max_demand_abs)
                if demand_val != 0:
                    temp_demands[loc_id][prod] = demand_val
                    product_global_demands[prod] += demand_val

    # Adjust last location's demand to balance totals (simple balancing heuristic)
    if num_locations > 0:
        last_loc_id = f"l{num_locations}"
        for prod in products_list:
            if product_global_demands[prod] != 0:
                temp_demands[last_loc_id][prod] -= product_global_demands[prod]
                # Reset global demand after balancing
                product_global_demands[prod] = 0

    for i in range(num_locations):
        loc_id = f"l{i + 1}"
        loc = Location(id=loc_id, net_supply=temp_demands[loc_id], name=loc_id)
        locations_list.append(loc)

    # Check if total demand for each product is truly balanced
    for prod in products_list:
        total_net_supply = sum(loc.net_supply.get(prod, 0) for loc in locations_list)
        if total_net_supply != 0:
            # If not perfectly balanced (due to rounding/edge cases), this is where it would be
            # acknowledged. For this generator, we rely on the heuristic.
            pass

    # 4. Generate TransportLink instances
    transport_links_list = []
    current_link_id = 0
    for i in range(num_locations):
        for j in range(num_locations):
            if i == j:  # No links from a location to itself
                continue

            loc1 = locations_list[i]
            loc2 = locations_list[j]

            # Shuffle transport types to randomize which ones get chosen if max_links_per_pair < num_transport_types
            available_tts_for_pair = list(transport_types_list)
            random.shuffle(available_tts_for_pair)

            links_created_for_pair = 0
            for tt in available_tts_for_pair:
                if links_created_for_pair >= max_links_per_pair:
                    break  # Limit links per pair to max_links_per_pair distinct transport types

                # Check is implicitly handled by iterating through a shuffled list of unique transport types.
                # Each TT will be used at most once for a given (loc1, loc2) pair in this loop.

                link = TransportLink(
                    id=current_link_id,  # Assign a unique ID for each link
                    location_l1=loc1,
                    location_l2=loc2,
                    distance=round(random.uniform(min_dist, max_dist), 1),
                    transport_type=tt,
                )
                transport_links_list.append(link)
                current_link_id += 1
                links_created_for_pair += 1

    # 5. Create MultibatchingProblem Instance
    problem_instance = MultibatchingProblem(
        transport_types=transport_types_list,
        products=products_list,
        locations=locations_list,
        transport_links=transport_links_list,
    )

    return problem_instance
