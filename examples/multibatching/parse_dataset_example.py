#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Example demonstrating how to parse multibatching datasets."""

from discrete_optimization.multibatching.parser import (
    get_data_available,
    parse_file,
)


def main():
    print("=" * 70)
    print("Multibatching Dataset Parsing Example")
    print("=" * 70)

    # Get available datasets
    print("\n1. Finding available datasets...")
    try:
        datasets = get_data_available()
        print(f"   Found {len(datasets)} dataset(s):")
        for i, dataset_path in enumerate(datasets, 1):
            print(f"   {i}. {dataset_path}")
    except FileNotFoundError as e:
        print(f"   Error: {e}")
        print("\n   To download datasets, run:")
        print(
            "   >>> from discrete_optimization.datasets import fetch_data_from_multibatching"
        )
        print("   >>> fetch_data_from_multibatching()")
        return

    if not datasets:
        print("   No datasets found.")
        return

    # Parse the first dataset
    print(f"\n2. Parsing dataset: {datasets[0]}")
    # Note: The original code used scale factors of 1/10^4
    # Adjust these based on your data requirements
    problem = parse_file(
        datasets[0],
        scale_capacity=1.0 / 10**4,  # Scale down capacities
        scale_size=1.0 / 10**4,  # Scale down product sizes
        scale_co2=1.0,  # No scaling for CO2
    )

    # Display problem information
    print("\n3. Problem Statistics:")
    print(f"   - Locations: {problem.nb_locations}")
    print(f"   - Products: {problem.nb_products}")
    print(f"   - Transport types: {problem.nb_transport_types}")
    print(f"   - Transport links: {problem.nb_transport_links}")

    print("\n4. Transport Types:")
    for tt in problem.transport_types[:5]:  # Show first 5
        print(
            f"   - {tt.name}: capacity={tt.capacity}, cost={tt.cost}, emissions={tt.emissions}"
        )
    if len(problem.transport_types) > 5:
        print(f"   ... and {len(problem.transport_types) - 5} more")

    print("\n5. Products:")
    for prod in problem.products[:5]:  # Show first 5
        print(f"   - {prod.name}: size={prod.size}, value={prod.value}")
    if len(problem.products) > 5:
        print(f"   ... and {len(problem.products) - 5} more")

    print("\n6. Locations with supply/demand:")
    for loc in problem.locations[:5]:  # Show first 5
        if loc.net_supply:
            print(f"   - {loc.name}:")
            for prod, qty in list(loc.net_supply.items())[:3]:
                sign = "supply" if qty > 0 else "demand"
                print(f"      {prod.name}: {abs(qty)} ({sign})")
    if len(problem.locations) > 5:
        print(f"   ... and {len(problem.locations) - 5} more")

    print("\n" + "=" * 70)
    print("Dataset parsed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
