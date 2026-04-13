"""Example 4: Sequential Pipeline with Transformations.

This example demonstrates:
- Using TransformationSolver within SequentialMetasolver
- Combining transformations with sequential solving
- Warmstart across problem transformations
- Multi-stage optimization workflows

Use case: Hybrid pipeline:
1. Greedy on original problem (fast)
2. CP-SAT via transformation (better model)
3. Local search on original problem (polish)
"""

from discrete_optimization.datasets import get_data_home
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.sequential_metasolver import (
    SequentialMetasolver,
)
from discrete_optimization.generic_tools.transformation import TransformationSolver
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.transformations import RcpspToMultiskillTransformation

try:
    from discrete_optimization.rcpsp.solvers.greedy import GreedyRcpspSolver

    from discrete_optimization.rcpsp_multiskill.solvers.cpsat import CPSatMSRcpspSolver

    SOLVERS_AVAILABLE = True
except ImportError:
    SOLVERS_AVAILABLE = False


def main():
    """Run sequential pipeline with transformations."""
    print("=" * 80)
    print("Example 4: Sequential Pipeline with Transformations")
    print("=" * 80)

    if not SOLVERS_AVAILABLE:
        print("\nThis example requires Greedy and CP-SAT solvers.")
        print("Install with: pip install ortools")
        return

    # Load RCPSP instance
    data_folder = f"{get_data_home()}/rcpsp/BL"
    files = get_data_available(data_folder)
    rcpsp_problem = parse_file(files[0])

    print(f"\nProblem: {rcpsp_problem.n_jobs} tasks")

    print("\n" + "=" * 80)
    print("Pipeline Architecture")
    print("=" * 80)
    print("""
┌────────────────────┐
│  RCPSP Problem     │
└────────┬───────────┘
         │
         ├─→ Stage 1: Greedy (RCPSP)
         │   Fast initial solution
         │
         ├─→ Stage 2: TransformationSolver
         │   ├─ Transform: RCPSP → Multiskill
         │   ├─ Solve: CP-SAT on Multiskill
         │   └─ Back-transform: Multiskill → RCPSP
         │
         └─→ Result: RcpspSolution (improved)
    """)

    # Create transformation
    transformation = RcpspToMultiskillTransformation()

    # Build sequential pipeline
    print("\nBuilding pipeline...")

    metasolver = SequentialMetasolver(
        problem=rcpsp_problem,
        list_subbricks=[
            # Stage 1: Greedy on original RCPSP
            SubBrick(cls=GreedyRcpspSolver, kwargs={}),
            # Stage 2: CP-SAT via transformation
            # TransformationSolver is a SolverDO, so it works as a subbrick!
            SubBrick(
                cls=TransformationSolver,
                kwargs={
                    "transformation": transformation,
                    "solver_brick": SubBrick(
                        cls=CPSatMSRcpspSolver,
                        kwargs={"time_limit": 15},
                    ),
                },
            ),
        ],
    )

    print("  Stage 1: Greedy on RCPSP")
    print("  Stage 2: CP-SAT via RCPSPMultiskill transformation")

    # Solve
    print("\nExecuting pipeline...")
    print("-" * 80)
    result = metasolver.solve()
    print("-" * 80)

    # Analyze results
    print(f"\nResults:")
    print(f"  - Total solutions: {len(result)}")

    if len(result) > 0:
        best_solution = result.get_best_solution()
        best_fit = result.get_best_solution_fit()[1]

        # Verify all solutions are RCPSP solutions
        all_rcpsp = all(type(sol).__name__ == "RcpspSolution" for sol, _ in result)
        print(f"  - All solutions are RcpspSolution: {all_rcpsp}")

        makespan = max(
            details["end_time"] for details in best_solution.rcpsp_schedule.values()
        )

        print(f"\nBest solution:")
        print(f"  - Fitness: {best_fit}")
        print(f"  - Makespan: {makespan}")
        print(f"  - Feasible: {rcpsp_problem.satisfy(best_solution)}")
        print(f"  - Type: {type(best_solution).__name__}")

        # Show solution progression (if we tracked intermediate)
        print("\nSolution progression:")
        for i, (sol, fit) in enumerate(result):
            ms = max(d["end_time"] for d in sol.rcpsp_schedule.values())
            print(f"  Solution {i + 1}: makespan={ms}, fitness={fit}")

    print("\n" + "=" * 80)
    print("How It Works")
    print("=" * 80)
    print("""
1. SequentialMetasolver instantiates each subbrick:
   - Greedy: GreedyRcpspSolver(problem=rcpsp_problem)
   - Transform: TransformationSolver(
                    problem=rcpsp_problem,  # ← Same problem!
                    transformation=...,
                    solver_brick=...
                )

2. Greedy solver runs:
   - Produces RcpspSolution

3. TransformationSolver runs:
   - Receives RcpspSolution as warmstart
   - Transforms RCPSP → Multiskill (including warmstart!)
   - Solves Multiskill problem
   - Back-transforms Multiskill solutions → RCPSP
   - Returns RcpspSolution (improved)

4. All solutions remain in RCPSP space throughout!
    """)

    print("\n" + "=" * 80)
    print("Advanced: 3-Stage Pipeline")
    print("=" * 80)
    print("""
# Add a third stage for final polishing
metasolver = SequentialMetasolver(
    problem=rcpsp_problem,
    list_subbricks=[
        # Stage 1: Quick greedy
        SubBrick(cls=GreedyRcpspSolver, kwargs={}),

        # Stage 2: Transform and solve
        SubBrick(
            cls=TransformationSolver,
            kwargs={
                "transformation": RcpspToMultiskillTransformation(),
                "solver_brick": SubBrick(cls=CPSatMSRcpspSolver, kwargs={...})
            }
        ),

        # Stage 3: Local search on original problem
        SubBrick(cls=LocalSearchRcpspSolver, kwargs={"iterations": 1000})
    ]
)

# Each stage receives warmstart from previous stage!
# Transformations are transparent to the pipeline.
    """)

    print("\n" + "=" * 80)
    print("Key Takeaways:")
    print("1. TransformationSolver works seamlessly with SequentialMetasolver")
    print("2. Transformations are transparent - all solutions stay in original space")
    print("3. Warmstart works across transformations")
    print("4. Can mix regular solvers and transformation solvers")
    print("5. Build complex pipelines: greedy → transform → polish")
    print("=" * 80)


if __name__ == "__main__":
    main()
