#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Example: Using the transformation graph for analysis and path finding.

This example shows how to use the transformation graph programmatically to:
- Find transformation paths between problems
- Discover available solvers via transformations
- Analyze transformation quality
- Get detailed metadata about transformations
"""

from discrete_optimization.generic_tools.transformation.solver_discovery import (
    build_solver_accessibility_report,
    find_best_transformation_path,
)
from discrete_optimization.generic_tools.transformation.transformation_graph import (
    WeightingStrategy,
    build_transformation_graph,
)


def example_1_build_and_explore_graph():
    """Build the transformation graph and explore its structure."""
    print("=" * 80)
    print("Example 1: Building and Exploring the Transformation Graph")
    print("=" * 80)

    # Build the graph (auto-discovers all transformations in codebase)
    graph = build_transformation_graph()

    # Print summary
    graph.print_summary()

    # List all problems
    print(f"\nAll problems in the graph ({graph.graph.number_of_nodes()}):")
    for problem in sorted(graph.graph.nodes()):
        print(f"  - {problem}")

    # List all transformations
    print(f"\nAll transformations ({len(graph.transformations)}):")
    for (source, target), edge in sorted(graph.transformations.items()):
        exactness = (
            "exact"
            if edge.forward_exact
            else f"lossy (impact: {edge.max_impact.value})"
        )
        print(f"  - {source} → {target} ({exactness})")

    return graph


def example_2_find_paths():
    """Find paths between different problems."""
    print("\n" + "=" * 80)
    print("Example 2: Finding Transformation Paths")
    print("=" * 80)

    graph = build_transformation_graph()

    # Example 1: BinPack to RCPSP
    source = "BinPackProblem"
    target = "RcpspProblem"

    print(f"\nFinding paths from {source} to {target}:")

    # Try different strategies
    strategies = [
        (WeightingStrategy.UNIFORM, "Minimize hops"),
        (WeightingStrategy.PREFER_EXACT, "Prefer exact transformations"),
        (WeightingStrategy.BY_IMPACT, "Minimize information loss"),
    ]

    for strategy, description in strategies:
        path = graph.find_path(source, target, strategy)
        if path:
            print(f"\n  Strategy: {description}")
            print(f"    Path: {' → '.join(path.problem_sequence)}")
            print(f"    Hops: {len(path.transformations)}")
            print(f"    Weight: {path.total_weight:.2f}")
            print(f"    Exact: {path.is_exact}")

            # Show transformations
            for i, trans in enumerate(path.transformations, 1):
                print(f"      {i}. {trans.transformation_class.__name__}")
        else:
            print(f"\n  Strategy: {description}")
            print(f"    No path found")


def example_3_find_all_paths():
    """Find all possible paths between two problems."""
    print("\n" + "=" * 80)
    print("Example 3: Finding All Paths")
    print("=" * 80)

    graph = build_transformation_graph()

    source = "BinPackProblem"
    target = "FacilityProblem"

    print(f"\nFinding all paths from {source} to {target} (max length 3):")

    all_paths = graph.find_all_paths(source, target, max_length=3)

    if all_paths:
        print(f"\nFound {len(all_paths)} path(s):")
        for i, path in enumerate(all_paths, 1):
            exact_str = (
                "exact" if path.is_exact else f"lossy (weight: {path.total_weight:.2f})"
            )
            print(f"\n  Path {i} ({exact_str}):")
            print(f"    {' → '.join(path.problem_sequence)}")

            # Show transformations
            for j, trans in enumerate(path.transformations, 1):
                print(f"      {j}. {trans.transformation_class.__name__}")
    else:
        print(f"\n  No paths found")


def example_4_reachability():
    """Analyze what problems are reachable from a given problem."""
    print("\n" + "=" * 80)
    print("Example 4: Reachability Analysis")
    print("=" * 80)

    graph = build_transformation_graph()

    problems = ["BinPackProblem", "SalbpProblem", "RcpspProblem"]

    for problem in problems:
        if problem not in graph.graph.nodes():
            continue

        reachable = graph.get_reachable_problems(problem)

        print(f"\n{problem}:")
        print(f"  Can reach {len(reachable) - 1} other problems")
        print(f"  Reachable problems:")
        for target in sorted(reachable - {problem}):
            # Find path
            path = graph.find_path(problem, target, WeightingStrategy.PREFER_EXACT)
            if path:
                exact_str = "exact" if path.is_exact else "lossy"
                hops = len(path.transformations)
                print(
                    f"    - {target} ({hops} hop{'s' if hops > 1 else ''}, {exact_str})"
                )


def example_5_transformation_details():
    """Get detailed information about specific transformations."""
    print("\n" + "=" * 80)
    print("Example 5: Transformation Details and Metadata")
    print("=" * 80)

    graph = build_transformation_graph()

    # Pick a lossy transformation to examine
    if ("BinPackProblem", "SalbpProblem") in graph.transformations:
        edge = graph.transformations[("BinPackProblem", "SalbpProblem")]

        print(f"\nExamining: {edge}")
        print(f"\nTransformation class: {edge.transformation_class.__name__}")
        print(f"Exactness: {edge.forward_exact and edge.backward_exact}")

        # Get metadata
        metadata = edge.transformation_instance.get_forward_metadata()

        print(f"\nCompleteness: {metadata.completeness.value}")

        if metadata.losses:
            print(f"\nInformation Losses ({len(metadata.losses)}):")
            for loss in metadata.losses:
                print(f"\n  Loss: {loss.name}")
                print(f"    Type: {loss.loss_type.value}")
                print(f"    Impact: {loss.impact.value}")
                print(f"    Description: {loss.description}")
                print(f"    Reason: {loss.reason}")
                if loss.workaround:
                    print(f"    Workaround: {loss.workaround}")

        if metadata.use_cases:
            print(f"\nRecommended use cases:")
            for use_case in metadata.use_cases:
                print(f"  - {use_case}")

        if metadata.warnings:
            print(f"\nWarnings:")
            for warning in metadata.warnings:
                print(f"  ⚠ {warning}")


def example_6_solver_discovery():
    """Discover all solvers available for a problem."""
    print("\n" + "=" * 80)
    print("Example 6: Solver Discovery")
    print("=" * 80)

    graph = build_transformation_graph()

    problem = "BinPackProblem"

    print(f"\nDiscovering solvers for {problem}...")

    # Build accessibility report
    report = build_solver_accessibility_report(problem, graph, max_path_length=3)

    print(f"\nTotal solvers available: {report.total_solvers}")
    print(f"  - Direct solvers: {len(report.direct_solvers)}")
    print(f"  - Via transformations: {len(report.transformation_solvers)}")

    if report.direct_solvers:
        print(f"\nDirect solvers:")
        for solver in report.direct_solvers[:5]:  # Show first 5
            print(f"  - {solver.solver_name}")
        if len(report.direct_solvers) > 5:
            print(f"  ... and {len(report.direct_solvers) - 5} more")

    if report.transformation_solvers:
        print(f"\nSolvers via transformations:")

        # Group by target problem
        by_target = {}
        for solver in report.transformation_solvers:
            target = solver.problem_type
            if target not in by_target:
                by_target[target] = []
            by_target[target].append(solver)

        for target, solvers in sorted(by_target.items()):
            exact_count = sum(1 for s in solvers if s.path.is_exact)
            print(
                f"\n  Via {target} ({len(solvers)} solvers, {exact_count} exact paths):"
            )
            for solver in solvers[:3]:  # Show first 3
                exact_str = "exact" if solver.path.is_exact else "lossy"
                hops = len(solver.path.transformations)
                print(
                    f"    - {solver.solver_name} ({hops} hop{'s' if hops > 1 else ''}, {exact_str})"
                )
            if len(solvers) > 3:
                print(f"    ... and {len(solvers) - 3} more")


def example_7_best_path():
    """Find the best transformation path between problems."""
    print("\n" + "=" * 80)
    print("Example 7: Finding the Best Transformation Path")
    print("=" * 80)

    graph = build_transformation_graph()

    pairs = [
        ("BinPackProblem", "RcpspProblem"),
        ("SalbpProblem", "FacilityProblem"),
        ("RcpspProblem", "MultiskillRcpspProblem"),
    ]

    for source, target in pairs:
        if source not in graph.graph.nodes() or target not in graph.graph.nodes():
            continue

        print(f"\n{source} → {target}:")

        # Find best path (prefers exact, then shortest, then lowest weight)
        path = find_best_transformation_path(source, target, graph)

        if path:
            exact_str = (
                "exact" if path.is_exact else f"lossy (weight: {path.total_weight:.2f})"
            )
            print(f"  Best path ({exact_str}):")
            print(f"    {' → '.join(path.problem_sequence)}")

            # Show transformations
            for i, trans in enumerate(path.transformations, 1):
                print(f"      {i}. {trans.transformation_class.__name__}")

            # Show what's lost if lossy
            if not path.is_exact:
                print(f"\n  Information losses:")
                for edge in path.transformations:
                    if not edge.forward_exact:
                        metadata = edge.transformation_instance.get_forward_metadata()
                        for loss in metadata.losses:
                            print(f"    - {loss.name} ({loss.impact.value})")
        else:
            print(f"  No path available")


def main():
    """Run all examples."""
    print("=" * 80)
    print("Transformation Graph Analysis Examples")
    print("=" * 80)
    print("\nThese examples demonstrate programmatic use of the transformation graph.")
    print("For interactive exploration, use the web-based viewer:")
    print("  uv run python launch_viewer.py")
    print("=" * 80)

    # Run examples
    graph = example_1_build_and_explore_graph()
    example_2_find_paths()
    example_3_find_all_paths()
    example_4_reachability()
    example_5_transformation_details()
    example_6_solver_discovery()
    example_7_best_path()

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("""
The transformation graph enables:

1. Automatic discovery of all transformations in the codebase
2. Path finding between any two problems
3. Solver discovery (direct + via transformations)
4. Transformation quality analysis (exact vs lossy)
5. Detailed metadata about information losses

Use the programmatic API for:
- Automated solver selection
- Building custom transformation pipelines
- Validating transformation chains
- Analyzing problem relationships

Use the interactive viewer for:
- Visual exploration
- Quick reference
- Team collaboration
- Documentation
    """)

    print("\nFor interactive visualization:")
    print("  uv run python launch_viewer.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
