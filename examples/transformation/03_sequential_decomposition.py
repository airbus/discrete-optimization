"""Example 3: Sequential Decomposition with kwargs_from_solution.

This example demonstrates:
- Solving a problem in stages (e.g., times first, then resources)
- Using SubBrick.kwargs_from_solution to pass data between stages
- Integration with SequentialMetasolver
- Extracting information from intermediate solutions

Note: This is a conceptual example showing the pattern.
In practice, you would implement solvers that accept kwargs like
fixed_start_times, fixed_modes, etc.
"""

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.sequential_metasolver import (
    SequentialMetasolver,
)
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solution import RcpspSolution

try:
    from discrete_optimization.rcpsp.solvers.cpsat import CpSatRcpspSolver

    CPSAT_AVAILABLE = True
except ImportError:
    CPSAT_AVAILABLE = False
    print("CP-SAT solver not available")


# Example extractor functions
def extract_makespan(solution: RcpspSolution) -> int:
    """Extract makespan from solution to use as upper bound."""
    return max(details["end_time"] for details in solution.rcpsp_schedule.values())


def extract_start_times(solution: RcpspSolution) -> dict:
    """Extract start times from solution."""
    return {
        task: details["start_time"] for task, details in solution.rcpsp_schedule.items()
    }


def extract_modes(solution: RcpspSolution) -> list:
    """Extract mode assignments from solution."""
    return solution.rcpsp_modes


def main():
    """Run sequential decomposition example."""
    print("=" * 80)
    print("Example 3: Sequential Decomposition with kwargs_from_solution")
    print("=" * 80)

    if not CPSAT_AVAILABLE:
        print("\nThis example requires the CP-SAT solver.")
        print("Install with: pip install ortools")
        return

    # Load RCPSP instance
    files = get_data_available()
    file = [f for f in files if "j301_1" in f][0]
    rcpsp_problem = parse_file(file)

    print(f"\nProblem: {rcpsp_problem.n_jobs} tasks, {rcpsp_problem.resources_list}")

    print("\n" + "=" * 80)
    print("Decomposition Strategy (Conceptual)")
    print("=" * 80)
    print("""
In a full implementation, you would:

Stage 1: Fast Solver for Initial Solution
  - Quick solution
  - Extract key data (makespan, start times, modes, etc.)

Stage 2: Refined Solver with Fixed Constraints
  - Use extracted data as constraints
  - Focus on improving specific aspects
    """)

    # Simple example: just run CP-SAT multiple times with increasing time limits
    print("\nBuilding sequential pipeline (simplified)...")

    metasolver = SequentialMetasolver(
        problem=rcpsp_problem,
        list_subbricks=[
            # Stage 1: Quick solve
            SubBrick(cls=CpSatRcpspSolver, kwargs={"time_limit": 5}),
            # Stage 2: Longer solve (with warmstart from stage 1)
            SubBrick(cls=CpSatRcpspSolver, kwargs={"time_limit": 15}),
        ],
    )

    print(f"  - Stage 1: {CpSatRcpspSolver.__name__} (5s)")
    print(f"  - Stage 2: {CpSatRcpspSolver.__name__} (15s, warmstart)")

    # Solve
    print("\nExecuting sequential pipeline...")
    result = metasolver.solve()

    print(f"\nResults:")
    print(f"  - Total solutions found: {len(result)}")

    if len(result) > 0:
        best_solution = result.get_best_solution()
        best_fit = result.get_best_solution_fit()[1]

        makespan = max(
            details["end_time"] for details in best_solution.rcpsp_schedule.values()
        )

        print(f"  - Best fitness: {best_fit}")
        print(f"  - Best makespan: {makespan}")
        print(f"  - Feasible: {rcpsp_problem.satisfy(best_solution)}")

    # Demonstrate extractor functions
    print("\n" + "=" * 80)
    print("Extractor Function Examples")
    print("=" * 80)

    if len(result) > 0:
        solution = result.get_best_solution()

        print("\n1. Extract makespan:")
        makespan_val = extract_makespan(solution)
        print(f"   makespan = {makespan_val}")

        print("\n2. Extract start times:")
        start_times = extract_start_times(solution)
        print(f"   Showing first 3 tasks:")
        for task, start in list(start_times.items())[:3]:
            print(f"     {task}: {start}")

        print("\n3. Extract modes:")
        modes = extract_modes(solution)
        print(f"   modes = {modes[:5]}... ({len(modes)} total)")

    # Show how kwargs_from_solution would be used
    print("\n" + "=" * 80)
    print("Using kwargs_from_solution (Advanced Pattern)")
    print("=" * 80)
    print("""
Example: Fix start times from first stage, optimize resources in second stage

# Hypothetical solver that accepts fixed_start_times
class CpSatRcpspSolverWithFixedTimes(CpSatRcpspSolver):
    def __init__(self, problem, fixed_start_times=None, **kwargs):
        super().__init__(problem, **kwargs)
        self.fixed_start_times = fixed_start_times

    def init_model(self, **kwargs):
        super().init_model(**kwargs)
        if self.fixed_start_times:
            for task, start_time in self.fixed_start_times.items():
                self.model.Add(self.start_vars[task] == start_time)

# Usage in SequentialMetasolver:
metasolver = SequentialMetasolver(
    problem=problem,
    list_subbricks=[
        # Stage 1: Solve for times
        SubBrick(cls=FastTimeSolver, kwargs={}),

        # Stage 2: Fix times, optimize resources
        SubBrick(
            cls=CpSatRcpspSolverWithFixedTimes,
            kwargs={"time_limit": 30},
            kwargs_from_solution={
                "fixed_start_times": extract_start_times  # ← Extract from Stage 1!
            }
        )
    ]
)

# The extractor function is called automatically:
# stage1_solution = stage1.solve().get_best_solution()
# fixed_times = extract_start_times(stage1_solution)  # ← Automatic!
# stage2 = CpSatRcpspSolverWithFixedTimes(problem, fixed_start_times=fixed_times)
    """)

    print("\n" + "=" * 80)
    print("Key Takeaways:")
    print("1. SequentialMetasolver chains solvers with automatic warmstart")
    print("2. kwargs_from_solution extracts data from previous solutions")
    print("3. Extracted data passed as kwargs to next solver")
    print("4. Enables decomposition: solve part 1, fix it, solve part 2")
    print("5. Extractor functions are reusable across different pipelines")
    print("=" * 80)


if __name__ == "__main__":
    main()
