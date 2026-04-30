#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Example: Solve BinPack via SALBP transformation.

This example demonstrates:
- Transforming BinPack problem to SALBP (Assembly Line Balancing)
- Solving the transformed problem with SALBP solvers
- Back-transforming solutions to BinPack space
"""

from discrete_optimization.alb.salbp.solvers.cpsat import CpSatSalbpSolver
from discrete_optimization.binpack.parser import (
    get_data_available_bppc,
    parse_bin_packing_constraint_file,
)
from discrete_optimization.binpack.transformations import BinpackToSalbpTransformation
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.transformation import TransformationSolver


def main():
    """Run BinPack → SALBP transformation example."""
    print("=" * 80)
    print("Example: Solve BinPack via SALBP Transformation")
    print("=" * 80)
    # Load a small BinPack instance
    files = get_data_available_bppc()
    binpack_problem = parse_bin_packing_constraint_file(files[0])
    binpack_problem.incompatible_items = set()
    binpack_problem.has_constraint = False
    print(f"\nOriginal BinPack problem:")
    print(f"  - Items: {binpack_problem.nb_items}")
    print(f"  - Bin capacity: {binpack_problem.capacity_bin}")
    print(
        f"  - Total weight: {sum(item.weight for item in binpack_problem.list_items)}"
    )

    # Create transformation
    print("\nCreating transformation: BinPack → SALBP...")
    transformation = BinpackToSalbpTransformation()

    # Transform problem (for inspection)
    salbp_problem = transformation.transform_problem(binpack_problem)
    print(f"\nTransformed SALBP problem:")
    print(f"  - Tasks: {salbp_problem.number_of_tasks}")
    print(f"  - Cycle time (station capacity): {salbp_problem.cycle_time}")
    print(f"  - Precedence constraints: {len(salbp_problem.precedence)}")

    # Solve using TransformationSolver
    print("\nSolving with SALBP CP-SAT solver...")
    solver = TransformationSolver(
        transformation=transformation,
        source_problem=binpack_problem,
        solver_brick=SubBrick(cls=CpSatSalbpSolver, kwargs={"time_limit": 10}),
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
        print(f"\nBin utilization:")
        for bin_id in sorted(weights.keys()):
            print(f"  - Bin {bin_id}: {weights[bin_id]}/{binpack_problem.capacity_bin}")

    print("\n" + "=" * 80)
    print("How It Works")
    print("=" * 80)
    print("""
1. BinPack → SALBP Transformation:
   - Items (weights) → Tasks (processing times)
   - Bin capacity → Station cycle time
   - Minimize bins → Minimize stations
   - No precedence constraints

2. Solve SALBP problem with CP-SAT

3. Back-transform SALBP solution → BinPack solution
   - Station assignments → Bin assignments

4. All solutions are valid BinPack solutions!
    """)

    print("=" * 80)
    print("Key Insight:")
    print("BinPack and SALBP are structurally identical when there are no")
    print("precedence constraints. This transformation enables using powerful")
    print("SALBP solvers for bin packing problems!")
    print("=" * 80)


if __name__ == "__main__":
    main()
