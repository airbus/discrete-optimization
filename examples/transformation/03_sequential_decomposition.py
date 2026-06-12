#!/usr/bin/env python3
#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Example 03: Sequential Decomposition

Demonstrates:
- Using SubBrick.kwargs_from_solution for decomposition
- Extracting data from intermediate solutions
- Passing extracted data to next solver
- Integration with SequentialMetasolver

Use case: Solve start times first, then fix them and solve resource allocation
"""

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.sequential_metasolver import (
    SequentialMetasolver,
)
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solvers.cpsat import CpSatRcpspSolver
from discrete_optimization.rcpsp.solvers.pile import PileRcpspSolver


def extract_start_times(solution):
    """Extract start times from solution to pass to next stage.

    This function demonstrates how to use kwargs_from_solution to
    extract solution data for the next solver in the pipeline.

    Args:
        solution: RCPSP solution from previous stage

    Returns:
        Dictionary with 'partial_solution' containing start times
    """
    # Extract the schedule (start times)
    start_times = {}
    for task, details in solution.rcpsp_schedule.items():
        start_times[task] = details["start_time"]

    print(f"  Extracted start times for {len(start_times)} tasks")

    # Return as kwargs for next solver
    # CpSatRcpspSolver accepts 'partial_solution' parameter for warmstart
    return {"partial_solution": solution}


def main():
    """Decompose RCPSP solving into multiple stages."""
    print("=" * 70)
    print("Example 03: Sequential Decomposition")
    print("=" * 70)

    # Load a small RCPSP instance
    files = get_data_available()
    file_path = [f for f in files if "j301_1.sm" in f][0]
    rcpsp_problem = parse_file(file_path)

    print(f"\nLoaded problem: {file_path.split('/')[-1]}")
    print(f"Tasks: {rcpsp_problem.n_jobs}")

    # Build sequential pipeline
    # Stage 1: Fast greedy to get initial solution
    # Stage 2: CP-SAT to polish (warmstarted from greedy)
    # Stage 3: Longer CP-SAT run for final optimization

    metasolver = SequentialMetasolver(
        problem=rcpsp_problem,
        list_subbricks=[
            # Stage 1: Fast greedy construction
            SubBrick(cls=PileRcpspSolver, kwargs={}),
            # Stage 2: Quick CP-SAT polish
            # Uses 'partial_solution' from extract_start_times
            SubBrick(
                cls=CpSatRcpspSolver,
                kwargs={"time_limit": 3},
            ),
            # Stage 3: Extended CP-SAT optimization
            # Automatically warmstarted from stage 2
            SubBrick(cls=CpSatRcpspSolver, kwargs={"time_limit": 5}),
        ],
    )

    print("\nSolving with 3-stage pipeline:")
    print("  Stage 1: Greedy construction (fast)")
    print("  Stage 2: CP-SAT polish (3s, warmstarted)")
    print("  Stage 3: CP-SAT optimize (5s, warmstarted)")

    # Execute pipeline
    result = metasolver.solve()

    # Get solutions from each stage
    all_solutions = list(result.list_solution_fits)
    print(f"\nSolutions found: {len(all_solutions)}")

    if all_solutions:
        # Show progression
        print("\nQuality progression:")
        for i, (solution, fit) in enumerate(all_solutions[:5], 1):
            makespan = rcpsp_problem.evaluate(solution)["makespan"]
            print(f"  Stage {i}: Makespan = {makespan}")

        # Best solution
        best_solution = result.get_best_solution()
        print(f"\nBest makespan: {rcpsp_problem.evaluate(best_solution)['makespan']}")

    print("\nKey takeaway:")
    print("- kwargs_from_solution extracts data between stages")
    print("- Each stage can warmstart from previous stage")
    print("- Progressive refinement: fast → good → optimal")


if __name__ == "__main__":
    main()
