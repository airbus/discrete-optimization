#!/usr/bin/env python3
#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Example 05: Parallel Solving

Demonstrates:
- Using SolverGraph for DAG-based workflows
- Branching: parallel solver strategies
- Merging: combining results with "best" strategy
- Mixing transformations and direct solvers

Use case: Try multiple approaches in parallel and pick the best solution
"""

from discrete_optimization.generic_tools.graph_solver.solver_graph import (
    SolverGraph,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
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
    """Try multiple solver strategies in parallel."""
    print("=" * 70)
    print("Example 05: Parallel Solving with SolverGraph")
    print("=" * 70)

    # Load a small RCPSP instance
    files = get_data_available()
    file_path = [f for f in files if "j301_1.sm" in f][0]
    rcpsp_problem = parse_file(file_path)

    print(f"\nLoaded problem: {file_path.split('/')[-1]}")
    print(f"Tasks: {rcpsp_problem.n_jobs}")

    # Build parallel solving graph
    graph = SolverGraph(source_problem=rcpsp_problem)

    # Strategy 1: Direct greedy solver
    greedy_id = graph.add_solver(
        node_id="greedy_direct",
        solver_brick=SubBrick(cls=PileRcpspSolver, kwargs={}),
    )

    # Strategy 2: Transform to Multiskill, then solve, then back-transform
    transformation = RcpspToMultiskillTransformation()
    to_multiskill_id = graph.add_transformation(
        node_id="to_multiskill",
        transformation=transformation,
    )
    solve_multiskill_id = graph.add_solver(
        node_id="solve_multiskill",
        solver_brick=SubBrick(cls=CpSatMultiskillRcpspSolver, kwargs={"time_limit": 5}),
    )
    back_multiskill_id = graph.add_back_transform(
        node_id="back_multiskill",
        transformation=transformation,
        source_problem=rcpsp_problem,
    )

    # Strategy 3: Direct CP-SAT on RCPSP
    direct_cpsat_id = graph.add_solver(
        node_id="direct_cpsat",
        solver_brick=SubBrick(cls=CpSatRcpspSolver, kwargs={"time_limit": 5}),
    )

    # Merge node: Pick best solution from all strategies
    merge_id = graph.add_merge(node_id="merge", strategy="best")

    # Build DAG edges (from root to solvers, solvers to merge)
    # Strategy 1: root → greedy → merge
    graph.add_edge("root", greedy_id)
    graph.add_edge(greedy_id, merge_id)

    # Strategy 2: root → transform → solve → back-transform → merge
    graph.add_edge("root", to_multiskill_id)
    graph.add_edge(to_multiskill_id, solve_multiskill_id)
    graph.add_edge(solve_multiskill_id, back_multiskill_id)
    graph.add_edge(back_multiskill_id, merge_id)

    # Strategy 3: root → cpsat → merge
    graph.add_edge("root", direct_cpsat_id)
    graph.add_edge(direct_cpsat_id, merge_id)

    print("\nSolver graph structure:")
    print("  Branch 1: Greedy RCPSP ─────────────────┐")
    print("  Branch 2: RCPSP → Multiskill → CP-SAT ──┼─→ Merge → Best")
    print("  Branch 3: CP-SAT RCPSP ─────────────────┘")

    # Execute graph
    print("\nExecuting parallel strategies...")
    result = graph.run()

    # Get best solution across all strategies
    if len(result) > 0:
        best_solution = result.get_best_solution()
        best_makespan = rcpsp_problem.evaluate(best_solution)["makespan"]

        print(f"\nResults from parallel execution:")
        print(f"  Total solutions found: {len(result)}")
        print(f"  Best makespan: {best_makespan}")

        # Show breakdown by strategy (if node info available)
        all_solutions = list(result.list_solution_fits)
        print(f"\n  All makespans: ", end="")
        makespans = [
            rcpsp_problem.evaluate(sol)["makespan"] for sol, _ in all_solutions
        ]
        print(f"{makespans}")

        assert rcpsp_problem.satisfy(best_solution)
        print("\n✓ Best solution is valid in RCPSP problem space")
    else:
        print("\nNo solutions found")

    print("\nKey takeaway:")
    print("- SolverGraph enables parallel exploration of strategies")
    print("- Mix direct solvers and transformations")
    print("- MergeNode automatically selects best solution")
    print("- DAG structure allows complex workflows")


if __name__ == "__main__":
    main()
