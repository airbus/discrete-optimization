#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Example: Solve Workforce Allocation via Coloring transformation.

This example demonstrates:
1. Loading a workforce allocation problem from benchmark
2. Transforming to ColoringProblem using composed transformation:
   - Allocation → ListColoring (direct encoding)
   - ListColoring → Coloring (dummy nodes encoding)
3. Solving with CP-SAT Coloring solver
4. Comparing with direct ListColoring approach
"""

from discrete_optimization.coloring.solvers.cpsat import CpSatColoringSolver
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.transformation import TransformationSolver
from discrete_optimization.workforce.allocation.parser import (
    parse_to_allocation_problem,
)
from discrete_optimization.workforce.allocation.transformations import (
    WorkforceAllocationToColoringTransformation,
)
from discrete_optimization.workforce.scheduling.parser import get_data_available


def solve_with_coloring():
    """Example 3: Compare both transformation approaches."""
    print("\n" + "=" * 80)
    print("Example 3: Comparison of both approaches")
    print("=" * 80)

    # Load problem
    files = get_data_available()
    allocation_problem = parse_to_allocation_problem(files[0])
    allocation_problem.allocation_additional_constraint.same_allocation = None
    trans1 = WorkforceAllocationToColoringTransformation()
    solver1 = TransformationSolver(
        transformation=trans1,
        source_problem=allocation_problem,
        solver_brick=SubBrick(cls=CpSatColoringSolver, kwargs={}),
    )
    result1 = solver1.solve()
    if len(result1) > 0:
        sol1 = result1.get_best_solution()
        eval1 = allocation_problem.evaluate(sol1)
        teams1 = len(set(sol1.allocation))
        print(f"    - Teams used: {teams1}")
        print(f"    - Feasible: {allocation_problem.satisfy(sol1)}")
        print(f"    - Evaluation: {eval1}")
    else:
        print(f"    - No solution found")
    print(f"\n  Key Differences:")


def main():
    """Run all examples."""
    print("Workforce Allocation to Coloring/ListColoring Examples")
    print("=" * 80)
    solve_with_coloring()


if __name__ == "__main__":
    main()
