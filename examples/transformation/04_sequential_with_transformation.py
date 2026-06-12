#!/usr/bin/env python3
#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Example 04: Sequential Solving and Transformation

Demonstrates:
- Sequential pipeline with warmstart
- Using TransformationSolver separately
- Comparing direct solving vs. transformation-based solving
- Integration patterns for transformations

Use case: Compare sequential pipeline vs. transformation solver
"""

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.sequential_metasolver import (
    SequentialMetasolver,
)
from discrete_optimization.generic_tools.transformation.transformation_solver import (
    TransformationSolver,
)
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solvers.cpsat import CpSatRcpspSolver
from discrete_optimization.rcpsp.solvers.pile import PileRcpspSolver
from discrete_optimization.rcpsp.transformations.to_multiskill import (
    RcpspToMultiskillTransformation,
)
from discrete_optimization.rcpsp_multiskill.solvers.cpsat import (
    CpSatMultiskillRcpspSolver,
)


def main():
    """Multi-stage pipeline with transformation."""
    print("=" * 70)
    print("Example 04: Sequential with Transformation")
    print("=" * 70)

    # Load a small RCPSP instance
    files = get_data_available()
    file_path = [f for f in files if "j301_1.sm" in f][0]
    rcpsp_problem = parse_file(file_path)

    print(f"\nLoaded problem: {file_path.split('/')[-1]}")
    print(f"Tasks: {rcpsp_problem.n_jobs}")

    # Create transformation
    transformation = RcpspToMultiskillTransformation()

    # Build pipeline:
    # 1. Greedy RCPSP (fast construction)
    # 2. CP-SAT RCPSP (polish with warmstart from greedy)

    # Note: TransformationSolver warmstart support is limited,
    # so we demonstrate a simpler 2-stage pipeline here
    metasolver = SequentialMetasolver(
        problem=rcpsp_problem,
        list_subbricks=[
            # Stage 1: Fast greedy in RCPSP
            SubBrick(cls=PileRcpspSolver, kwargs={}),
            # Stage 2: CP-SAT polish
            SubBrick(cls=CpSatRcpspSolver, kwargs={"time_limit": 5}),
        ],
    )

    # Alternative: Use TransformationSolver standalone (not in pipeline)
    # This demonstrates TransformationSolver usage without warmstart complexity

    print("\nSolving with sequential pipeline:")
    print("  Stage 1: Greedy RCPSP")
    print("  Stage 2: CP-SAT RCPSP (warmstarted)")

    # Execute pipeline
    result = metasolver.solve()

    # Get all solutions
    all_solutions = list(result.list_solution_fits)
    print(f"\nSolutions found: {len(all_solutions)}")

    if all_solutions:
        # Show progression
        print("\nQuality progression:")
        stages = ["Greedy RCPSP", "CP-SAT RCPSP"]
        for i, ((solution, fit), stage_name) in enumerate(
            zip(all_solutions[:2], stages), 1
        ):
            makespan = rcpsp_problem.evaluate(solution)["makespan"]
            print(f"  Stage {i} ({stage_name}): Makespan = {makespan}")

        # Best solution from sequential
        best_makespan_seq = rcpsp_problem.evaluate(result.get_best_solution())[
            "makespan"
        ]
        print(f"\nBest makespan (sequential): {best_makespan_seq}")

    # Now demonstrate TransformationSolver usage separately
    print("\n--- Solving via transformation (Multiskill) ---")
    trans_solver = TransformationSolver(
        transformation=transformation,
        source_problem=rcpsp_problem,
        solver_brick=SubBrick(cls=CpSatMultiskillRcpspSolver, kwargs={"time_limit": 5}),
    )
    trans_result = trans_solver.solve()

    if len(trans_result) > 0:
        trans_solution = trans_result.get_best_solution()
        trans_makespan = rcpsp_problem.evaluate(trans_solution)["makespan"]
        print(f"Makespan via Multiskill transformation: {trans_makespan}")
        assert rcpsp_problem.satisfy(trans_solution)
        print("✓ Transformation solution is valid in RCPSP problem space")

    print("\nKey takeaway:")
    print("- Sequential pipelines enable progressive improvement")
    print("- TransformationSolver provides alternative formulations")
    print("- Compare multiple approaches to find best strategy")
    print("- All solutions stay in original problem space")


if __name__ == "__main__":
    main()
