#!/usr/bin/env python3
#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Example 01: Simple Transformation

Demonstrates:
- Creating a transformation (RCPSP → RCPSPMultiskill)
- Using TransformationSolver
- Automatic back-transformation of solutions
- Solutions remain in original problem space
"""

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.transformation.transformation_solver import (
    TransformationSolver,
)
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.transformations.to_multiskill import (
    RcpspToMultiskillTransformation,
)
from discrete_optimization.rcpsp_multiskill.solvers.cpsat import (
    CpSatMultiskillRcpspSolver,
)


def main():
    """Solve RCPSP by transforming to RCPSPMultiskill."""
    print("=" * 70)
    print("Example 01: Simple Transformation (RCPSP → Multiskill)")
    print("=" * 70)

    # Load a small RCPSP instance
    files = get_data_available()
    file_path = [f for f in files if "j301_1.sm" in f][0]
    rcpsp_problem = parse_file(file_path)

    print(f"\nLoaded problem: {file_path.split('/')[-1]}")
    print(f"Tasks: {rcpsp_problem.n_jobs}")
    print(f"Resources: {rcpsp_problem.resources_list}")

    # Create transformation
    transformation = RcpspToMultiskillTransformation()

    # Check transformation metadata
    metadata = transformation.get_forward_metadata()
    print(f"\nTransformation type: {metadata.completeness}")
    print(f"Use cases: {metadata.use_cases[0]}")

    # Create TransformationSolver
    # This solver:
    # 1. Transforms RCPSP → Multiskill
    # 2. Solves in Multiskill space with CP-SAT
    # 3. Automatically back-transforms solution to RCPSP
    solver = TransformationSolver(
        transformation=transformation,
        source_problem=rcpsp_problem,
        solver_brick=SubBrick(cls=CpSatMultiskillRcpspSolver, kwargs={"time_limit": 5}),
    )

    print("\nSolving...")
    result = solver.solve()

    # Solution is automatically back-transformed to RCPSP space
    solution = result.get_best_solution()
    print(f"\nSolution type: {type(solution).__name__}")
    print(f"Makespan: {rcpsp_problem.evaluate(solution)['makespan']}")

    # Verify solution is valid in original problem space
    assert rcpsp_problem.satisfy(solution)
    print("✓ Solution is valid in RCPSP problem space")

    print("\nKey takeaway:")
    print("- TransformationSolver handles all transformation logic")
    print("- Solution is automatically in original problem space")
    print("- No manual transformation needed!")


if __name__ == "__main__":
    main()
