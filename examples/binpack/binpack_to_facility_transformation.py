#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Example: Solve BinPack via Facility Location transformation.

This example demonstrates:
- Transforming BinPack problem to Facility Location
- Solving the transformed problem with Facility Location solvers
- Back-transforming solutions to BinPack space
"""

from discrete_optimization.binpack.parser import (
    get_data_available_bppc,
    parse_bin_packing_constraint_file,
)
from discrete_optimization.binpack.transformations import (
    BinpackToFacilityTransformation,
)
from discrete_optimization.facility.solvers.greedy import GreedyFacilitySolver
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.transformation import TransformationSolver


def main():
    """Run BinPack → Facility Location transformation example."""
    print("=" * 80)
    print("Example: Solve BinPack via Facility Location Transformation")
    print("=" * 80)
    # Load a small BinPack instance
    files = get_data_available_bppc()
    binpack_problem = parse_bin_packing_constraint_file(files[0])
    binpack_problem.has_constraint = False
    binpack_problem.incompatible_items = set()

    print(f"\nOriginal BinPack problem:")
    print(f"  - Items: {binpack_problem.nb_items}")
    print(f"  - Bin capacity: {binpack_problem.capacity_bin}")
    print(
        f"  - Total weight: {sum(item.weight for item in binpack_problem.list_items)}"
    )

    # Create transformation with setup cost = 1 per bin
    print("\nCreating transformation: BinPack → Facility Location...")
    transformation = BinpackToFacilityTransformation(setup_cost_per_bin=1.0)

    # Transform problem (for inspection)
    facility_problem = transformation.transform_problem(binpack_problem)
    print(f"\nTransformed Facility Location problem:")
    print(f"  - Customers (items): {facility_problem.customer_count}")
    print(f"  - Facilities (potential bins): {facility_problem.facility_count}")
    print(
        f"  - Facility capacity: {facility_problem.facilities[0].capacity if facility_problem.facilities else 'N/A'}"
    )
    print(
        f"  - Setup cost per facility: {facility_problem.facilities[0].setup_cost if facility_problem.facilities else 'N/A'}"
    )

    # Solve using TransformationSolver with Greedy
    print("\nSolving with Greedy Facility Location solver...")
    solver = TransformationSolver(
        transformation=transformation,
        source_problem=binpack_problem,
        solver_brick=SubBrick(cls=GreedyFacilitySolver, kwargs={}),
    )

    result = solver.solve()

    # Analyze results
    print(f"\nResults:")
    print(f"  - Solutions found: {len(result)}")

    if len(result) > 0:
        best_solution = result.get_best_solution()
        best_fit = result.get_best_solution_fit()[1]

        print(f"\nBest solution:")
        print(f"  - Fitness: {best_fit}")
        print(f"  - Solution type: {type(best_solution).__name__}")
        print(f"  - Feasible: {binpack_problem.satisfy(best_solution)}")

        # Evaluate in BinPack space
        eval_result = binpack_problem.evaluate(best_solution)
        print(f"  - Number of bins used: {eval_result['nb_bins']}")
        print(f"  - Penalty: {eval_result['penalty']}")

        # Check bin weights
        weights = binpack_problem.compute_weights(best_solution)
        print(f"\nBin utilization (showing first 10):")
        for bin_id in sorted(list(weights.keys())[:10]):
            util_pct = (weights[bin_id] / binpack_problem.capacity_bin) * 100
            print(
                f"  - Bin {bin_id}: {weights[bin_id]}/{binpack_problem.capacity_bin} ({util_pct:.1f}%)"
            )
        if len(weights) > 10:
            print(f"  ... and {len(weights) - 10} more bins")

    print("\n" + "=" * 80)
    print("How It Works")
    print("=" * 80)
    print("""
1. BinPack → Facility Location Transformation:
   - Items (weights) → Customers (demands)
   - Bins (capacity) → Facilities (capacity)
   - Opening a bin → Setup cost (minimized)
   - No spatial locations (all at origin)
   - No assignment costs (distance = 0)

2. Solve Facility Location with Greedy solver

3. Back-transform Facility solution → BinPack solution
   - Facility assignments → Bin assignments

4. All solutions are valid BinPack solutions!
    """)

    print("=" * 80)
    print("Key Insight:")
    print("Facility Location generalizes Bin Packing by adding:")
    print("  - Setup costs (bin opening cost)")
    print("  - Assignment costs (spatial distance)")
    print("When assignment costs = 0, it reduces to Bin Packing!")
    print("=" * 80)


if __name__ == "__main__":
    main()
