"""Example 5: Parallel Solving with SolverGraph.

This example demonstrates:
- Branching: Try multiple solver strategies in parallel
- Merging: Combine results with "best" strategy
- DAG-based workflows vs sequential pipelines
- Mixing transformations and direct solvers

Use case: Solve RCPSP using parallel strategies:
1. Direct CP-SAT on RCPSP
2. Transform to Multiskill, then CP-SAT
3. Merge best results from both approaches
"""

from discrete_optimization.generic_tools.graph_solver import SolverGraph
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solvers.cpsat import CpSatRcpspSolver
from discrete_optimization.rcpsp.transformations import RcpspToMultiskillTransformation
from discrete_optimization.rcpsp_multiskill.solvers.cpsat import (
    CpSatMultiskillRcpspSolver,
)


def main():
    """Run parallel solving with SolverGraph."""
    print("=" * 80)
    print("Example 5: Parallel Solving with SolverGraph")
    print("=" * 80)

    # Load RCPSP instance
    files = get_data_available()
    file = [f for f in files if "j301_1" in f][0]
    rcpsp_problem = parse_file(file)

    print(f"\nProblem: {rcpsp_problem.n_jobs} tasks")

    print("\n" + "=" * 80)
    print("Graph Architecture")
    print("=" * 80)
    print("""
                    ┌─────────────┐
                    │ RCPSP       │
                    │ (root)      │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │                         │
              ▼                         ▼
    ┌─────────────────┐      ┌─────────────────┐
    │ CP-SAT Direct   │      │ Transform to    │
    │ (solver1)       │      │ Multiskill      │
    └────────┬────────┘      │ (transform)     │
             │               └────────┬─────────┘
             │                        │
             │                        ▼
             │               ┌─────────────────┐
             │               │ CP-SAT on       │
             │               │ Multiskill      │
             │               │ (solver2)       │
             │               └────────┬─────────┘
             │                        │
             │                        ▼
             │               ┌─────────────────┐
             │               │ Back-transform  │
             │               │ to RCPSP        │
             │               │ (back_transform)│
             │               └────────┬─────────┘
             │                        │
             └────────────┬───────────┘
                          ▼
                  ┌───────────────┐
                  │ Merge Best    │
                  │ (merge)       │
                  └───────────────┘
    """)

    # Create transformation
    transformation = RcpspToMultiskillTransformation()

    # Build the solver graph
    print("\nBuilding SolverGraph...")
    graph = SolverGraph(source_problem=rcpsp_problem)

    # Branch 1: Direct CP-SAT on RCPSP
    graph.add_solver(
        "solver1", SubBrick(cls=CpSatRcpspSolver, kwargs={"time_limit": 10})
    )
    graph.add_edge("root", "solver1")

    # Branch 2: Transform → Solve → Back-transform
    graph.add_transformation("transform", transformation)
    graph.add_solver(
        "solver2",
        SubBrick(cls=CpSatMultiskillRcpspSolver, kwargs={"time_limit": 10}),
    )
    graph.add_back_transform("back_transform", transformation, rcpsp_problem)

    graph.add_edge("root", "transform")
    graph.add_edge("transform", "solver2")
    graph.add_edge("solver2", "back_transform")

    # Merge results from both branches
    graph.add_merge("merge", strategy="best")
    graph.add_edge("solver1", "merge")
    graph.add_edge("back_transform", "merge")

    # Visualize the graph
    print("\n" + graph.visualize())

    # Execute the graph
    print("\n" + "=" * 80)
    print("Executing Graph")
    print("=" * 80)
    result = graph.run()

    # Analyze results
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)
    print(f"Total solutions: {len(result)}")

    if len(result) > 0:
        best_solution = result.get_best_solution()
        print(result.list_solution_fits)
        best_fit = result.get_best_solution_fit()[1]

        makespan = max(
            details["end_time"] for details in best_solution.rcpsp_schedule.values()
        )

        print(f"\nBest solution:")
        print(f"  - Fitness: {best_fit}")
        print(f"  - Makespan: {makespan}")
        print(f"  - Feasible: {rcpsp_problem.satisfy(best_solution)}")
        print(f"  - Type: {type(best_solution).__name__}")

        # Show all solutions (from both branches)
        print(f"\nAll solutions (top 5):")
        for i, (sol, fit) in enumerate(list(result)[:5]):
            ms = max(d["end_time"] for d in sol.rcpsp_schedule.values())
            print(f"  Solution {i + 1}: makespan={ms}, fitness={fit}")

    print("\n" + "=" * 80)
    print("Advanced: More Complex Graphs")
    print("=" * 80)
    print("""
Example: Multiple transformations in parallel

graph = SolverGraph(problem)

# Branch 1: Transform to Multiskill
graph.add_transformation("to_ms", RcpspToMultiskillTransformation())
graph.add_solver("cpsat_ms", SubBrick(cls=CPSatMS, kwargs={}))
graph.add_back_transform("from_ms", transformation1, problem)
graph.add_edge("root", "to_ms")
graph.add_edge("to_ms", "cpsat_ms")
graph.add_edge("cpsat_ms", "from_ms")

# Branch 2: Transform to Preemptive
graph.add_transformation("to_preempt", RcpspToPreemptiveTransformation())
graph.add_solver("cpsat_preempt", SubBrick(cls=CPSatPreemptive, kwargs={}))
graph.add_back_transform("from_preempt", transformation2, problem)
graph.add_edge("root", "to_preempt")
graph.add_edge("to_preempt", "cpsat_preempt")
graph.add_edge("cpsat_preempt", "from_preempt")

# Branch 3: Direct MILP
graph.add_solver("milp", SubBrick(cls=GurobiRcpsp, kwargs={}))
graph.add_edge("root", "milp")

# Merge all three approaches
graph.add_merge("final_merge", strategy="all")
graph.add_edge("from_ms", "final_merge")
graph.add_edge("from_preempt", "final_merge")
graph.add_edge("milp", "final_merge")

result = graph.run(time_limit=60)
    """)

    print("\n" + "=" * 80)
    print("SolverGraph vs SequentialMetasolver")
    print("=" * 80)
    print("""
SequentialMetasolver:
- Linear pipeline: solver1 → solver2 → solver3
- Each solver warm-starts from previous
- Good for: iterative improvement

SolverGraph:
- Arbitrary DAG: branching, merging, parallel paths
- More complex workflows
- Good for: trying multiple strategies, combining approaches

Both can use TransformationSolver for problem transformations!
    """)

    print("\n" + "=" * 80)
    print("Key Takeaways:")
    print("1. SolverGraph enables parallel solver strategies")
    print("2. Branching: try multiple approaches simultaneously")
    print("3. Merging: combine best results from different paths")
    print("4. Can mix transformations and direct solvers in the same graph")
    print("5. All solutions automatically back-transformed to original problem")
    print("=" * 80)


if __name__ == "__main__":
    main()
