#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Demonstrate TSP transformations to TSPTW and GPDP.

This example demonstrates:
1. TSP → TSPTW transformation (exact, adds wide time windows)
2. TSP → GPDP transformation (exact, single vehicle no constraints)
3. Round-trip solution transformation verification
4. Solving TSP via transformed problems using different solvers
"""

import logging

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.transformation.transformation_solver import (
    TransformationSolver,
)
from discrete_optimization.tsp.parser import get_data_available, parse_file
from discrete_optimization.tsp.transformations import (
    TspToGpdpTransformation,
    TspToTsptwTransformation,
)
from discrete_optimization.tsptw.solvers.cpsat import CpSatTSPTWSolver

# Optional GPDP solver import (requires OR-Tools)
try:
    from ortools.constraint_solver import routing_enums_pb2

    from discrete_optimization.gpdp.solvers.ortools_routing import (
        OrtoolsGpdpSolver,
        ParametersCost,
    )

    GPDP_AVAILABLE = True
except ImportError:
    GPDP_AVAILABLE = False
    print("⚠️  OR-Tools not available - GPDP transformation test will be skipped")

logging.basicConfig(level=logging.WARNING)


def solve_tsp_via_tsptw(time_limit: int = 10):
    """Solve TSP via TSPTW transformation using CP-SAT solver.

    Args:
        time_limit: Time limit in seconds for the solver
    """
    print("\n" + "=" * 80)
    print("Example 1: Solving TSP via TSPTW Transformation")
    print("=" * 80)

    # Load TSP problem (small instance for quick testing)
    files = get_data_available()
    files = [f for f in files if "tsp_51_1" in f]
    if not files:
        print("⚠️  tsp_51_1 not found, using first available dataset")
        files = get_data_available()
    tsp_problem = parse_file(files[0], start_index=0, end_index=0)
    print(f"TSP Problem: {tsp_problem.node_count} nodes")

    # Create transformation
    transformation = TspToTsptwTransformation(horizon=10000)
    print("\n1. Transforming TSP → TSPTW...")
    tsptw_problem = transformation.transform_problem(tsp_problem)
    print(f"   ✓ TSPTW problem created: {tsptw_problem.nb_nodes} nodes")
    print(f"   ✓ Time windows: all [0, {tsptw_problem.time_windows[0][1]}] (relaxed)")

    # Solve via transformation
    print(f"\n2. Solving TSP via TSPTW (CP-SAT, {time_limit}s)...")
    solver = TransformationSolver(
        transformation=transformation,
        source_problem=tsp_problem,
        solver_brick=SubBrick(
            cls=CpSatTSPTWSolver,
            kwargs={"time_limit": time_limit},
        ),
    )
    solver.init_model()
    result = solver.solve()
    best_solution = result.get_best_solution()

    if best_solution:
        print(f"   ✓ Solution found!")
        print(f"     - Tour length: {-best_solution.length:.2f}")
        print(f"     - Feasible: {tsp_problem.satisfy(best_solution)}")

        # Verify round-trip transformation
        print("\n3. Verifying round-trip transformation...")
        tsptw_sol = transformation.forward_transform_solution(
            best_solution, tsptw_problem
        )
        tsp_sol_back = transformation.back_transform_solution(tsptw_sol, tsp_problem)
        original_obj = tsp_problem.evaluate(best_solution)["length"]
        roundtrip_obj = tsp_problem.evaluate(tsp_sol_back)["length"]
        print(f"     - Original objective: {original_obj:.2f}")
        print(f"     - Round-trip objective: {roundtrip_obj:.2f}")
        print(f"     - Objective preserved: {abs(original_obj - roundtrip_obj) < 0.01}")
    else:
        print("   ✗ No solution found")

    print("\n✅ TSP → TSPTW example complete")
    return best_solution


def solve_tsp_via_gpdp(time_limit: int = 10):
    """Solve TSP via GPDP transformation using OR-Tools routing solver.

    Args:
        time_limit: Time limit in seconds for the solver
    """
    if not GPDP_AVAILABLE:
        print("\n⚠️  Skipping TSP → GPDP example (OR-Tools not available)")
        return None

    print("\n" + "=" * 80)
    print("Example 2: Solving TSP via GPDP Transformation")
    print("=" * 80)

    # Load TSP problem
    files = get_data_available()
    files = [f for f in files if "tsp_51_1" in f]
    if not files:
        print("⚠️  tsp_51_1 not found, using first available dataset")
        files = get_data_available()
    tsp_problem = parse_file(files[0], start_index=0, end_index=0)
    print(f"TSP Problem: {tsp_problem.node_count} nodes")

    # Create transformation
    transformation = TspToGpdpTransformation(compute_graph=True)
    print("\n1. Transforming TSP → GPDP...")
    gpdp_problem = transformation.transform_problem(tsp_problem)
    print(f"   ✓ GPDP problem created: {gpdp_problem.number_vehicle} vehicle")
    print(f"   ✓ Customers: {len(gpdp_problem.nodes_transportation)}")

    # Solve via transformation
    print(f"\n2. Solving TSP via GPDP (OR-Tools, {time_limit}s)...")
    solver = TransformationSolver(
        transformation=transformation,
        source_problem=tsp_problem,
        solver_brick=SubBrick(
            cls=OrtoolsGpdpSolver,
            kwargs={"factor_multiplier_time": 100},
        ),
    )

    # Explicit init_model call for GPDP solver
    solver.init_model(
        one_visit_per_node=True,
        include_time_windows=False,  # TSP has no time constraints
        include_time_dimension=False,
        include_demand=False,
        include_pickup_and_delivery=False,
        local_search_metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC,
        first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
        time_limit=time_limit,
        parameters_cost=[ParametersCost(dimension_name="Distance", global_span=True)],
    )

    result = solver.solve()
    best_solution = result.get_best_solution()

    if best_solution:
        print(f"   ✓ Solution found!")
        print(f"     - Tour length: {-best_solution.length:.2f}")
        print(f"     - Feasible: {tsp_problem.satisfy(best_solution)}")

        # Verify round-trip transformation
        print("\n3. Verifying round-trip transformation...")
        gpdp_sol = transformation.forward_transform_solution(
            best_solution, gpdp_problem
        )
        tsp_sol_back = transformation.back_transform_solution(gpdp_sol, tsp_problem)
        original_obj = tsp_problem.evaluate(best_solution)["length"]
        roundtrip_obj = tsp_problem.evaluate(tsp_sol_back)["length"]
        print(f"     - Original objective: {original_obj:.2f}")
        print(f"     - Round-trip objective: {roundtrip_obj:.2f}")
        print(f"     - Objective preserved: {abs(original_obj - roundtrip_obj) < 0.01}")
    else:
        print("   ✗ No solution found")

    print("\n✅ TSP → GPDP example complete")
    return best_solution


def main(time_limit: int = 10):
    """Run TSP transformation examples.

    Args:
        time_limit: Time limit in seconds for each solver
    """
    print("\n" + "=" * 80)
    print("TSP Problem Transformation Examples")
    print("=" * 80)
    print("\nDemonstrating exact transformations:")
    print("  1. TSP → TSPTW (adds wide time windows)")
    print("  2. TSP → GPDP (single vehicle, no constraints)")

    solve_tsp_via_tsptw(time_limit=time_limit)
    solve_tsp_via_gpdp(time_limit=time_limit)

    print("\n" + "=" * 80)
    print("Summary:")
    print("  ✓ Both transformations are exact (no information loss)")
    print("  ✓ Round-trip transformations preserve objectives")
    print("  ✓ TSP can be solved via TSPTW and GPDP solvers")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
