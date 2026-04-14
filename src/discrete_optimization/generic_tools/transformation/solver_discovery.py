#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Solver discovery via transformation graph.

This module uses the transformation graph to discover all solvers
accessible for a given problem type, including those available
through problem transformations.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.transformation.transformation_graph import (
    TransformationGraph,
    TransformationPath,
    WeightingStrategy,
    build_transformation_graph,
)


@dataclass
class SolverOption:
    """Represents a solver option for a problem."""

    problem_type: str
    solver_class: type[SolverDO]
    solver_name: str
    path: Optional[TransformationPath] = None  # None if direct solver
    is_direct: bool = True  # True if no transformation needed

    @property
    def access_method(self) -> str:
        """Get access method description."""
        if self.is_direct:
            return "Direct"
        else:
            return f"Via {' → '.join(self.path.problem_sequence)}"

    def __str__(self) -> str:
        """String representation."""
        if self.is_direct:
            return f"{self.solver_name} (Direct)"
        else:
            path_str = " → ".join(self.path.problem_sequence)
            exact_str = (
                " [exact]"
                if self.path.is_exact
                else f" [lossy, weight: {self.path.total_weight:.1f}]"
            )
            return f"{self.solver_name} via {path_str}{exact_str}"


@dataclass
class SolverAccessibilityReport:
    """Report of all solvers accessible for a problem type."""

    problem_type: str
    direct_solvers: list[SolverOption] = field(default_factory=list)
    transformation_solvers: list[SolverOption] = field(default_factory=list)

    @property
    def total_solvers(self) -> int:
        """Total number of solvers."""
        return len(self.direct_solvers) + len(self.transformation_solvers)

    def print_report(self) -> None:
        """Print human-readable report."""
        print("=" * 80)
        print(f"Solver Accessibility Report: {self.problem_type}")
        print("=" * 80)

        print(f"\nTotal solvers available: {self.total_solvers}")
        print(f"  - Direct solvers: {len(self.direct_solvers)}")
        print(f"  - Via transformations: {len(self.transformation_solvers)}")

        if self.direct_solvers:
            print("\n" + "-" * 80)
            print("DIRECT SOLVERS (No transformation needed)")
            print("-" * 80)
            for solver in self.direct_solvers:
                print(f"  • {solver.solver_name}")

        if self.transformation_solvers:
            print("\n" + "-" * 80)
            print("TRANSFORMATION-BASED SOLVERS")
            print("-" * 80)

            # Group by target problem
            by_target = {}
            for solver in self.transformation_solvers:
                target = solver.problem_type
                if target not in by_target:
                    by_target[target] = []
                by_target[target].append(solver)

            for target, solvers in sorted(by_target.items()):
                print(f"\n  Via {target}:")
                for solver in solvers:
                    path_str = " → ".join(solver.path.problem_sequence)
                    exact_str = " [exact]" if solver.path.is_exact else " [lossy]"
                    print(f"    • {solver.solver_name}{exact_str}")
                    print(f"      Path: {path_str}")


def discover_solvers_for_problem(
    problem_type: str,
) -> dict[str, list[type[SolverDO]]]:
    """Discover all solver classes for a problem type.

    Args:
        problem_type: Problem class name (e.g., "BinPackProblem")

    Returns:
        Dict mapping solver names to solver classes

    """
    solvers = {}

    # Convert problem name to module name
    # e.g., BinPackProblem -> binpack
    module_name = _problem_name_to_module(problem_type)

    if not module_name:
        return solvers

    # Try to import solvers module
    solvers_module_path = f"discrete_optimization.{module_name}.solvers"

    try:
        base_solvers_module = importlib.import_module(solvers_module_path)
        solvers_path = Path(base_solvers_module.__file__).parent

        # Scan all solver modules
        for solver_file in solvers_path.glob("*.py"):
            if solver_file.name.startswith("_"):
                continue

            solver_module_path = f"{solvers_module_path}.{solver_file.stem}"

            try:
                solver_module = importlib.import_module(solver_module_path)

                # Find solver classes
                for name, obj in inspect.getmembers(solver_module):
                    if (
                        inspect.isclass(obj)
                        and issubclass(obj, SolverDO)
                        and obj != SolverDO
                        and not inspect.isabstract(obj)
                    ):
                        solvers[name] = obj

            except ImportError:
                # Skip modules with missing dependencies
                pass
            except Exception as e:
                # Skip problematic modules
                pass

    except ImportError:
        # No solvers module for this problem
        pass

    return solvers


def _problem_name_to_module(problem_type: str) -> Optional[str]:
    """Convert problem class name to module name.

    Args:
        problem_type: Problem class name (e.g., "BinPackProblem")

    Returns:
        Module name (e.g., "binpack") or None

    """
    # Special cases
    special_cases = {
        "BinPackProblem": "binpack",
        "SalbpProblem": "salbp",
        "RcpspProblem": "rcpsp",
        "MultiskillRcpspProblem": "rcpsp_multiskill",
        "FJobShopProblem": "fjsp",
        "JobShopProblem": "jsp",
        "FacilityProblem": "facility",
        "KnapsackProblem": "knapsack",
        "TspProblem": "tsp",
        "VrpProblem": "vrp",
    }

    if problem_type in special_cases:
        return special_cases[problem_type]

    # Try to infer from name
    # Remove "Problem" suffix and convert to lowercase
    if problem_type.endswith("Problem"):
        base = problem_type[:-7]
        # Convert CamelCase to snake_case
        import re

        module = re.sub(r"(?<!^)(?=[A-Z])", "_", base).lower()
        return module

    return None


def build_solver_accessibility_report(
    problem_type: str,
    transformation_graph: Optional[TransformationGraph] = None,
    max_path_length: int = 3,
) -> SolverAccessibilityReport:
    """Build comprehensive solver accessibility report.

    Args:
        problem_type: Problem class name
        transformation_graph: Optional pre-built graph (will build if None)
        max_path_length: Maximum transformation path length

    Returns:
        SolverAccessibilityReport

    """
    # Build transformation graph if not provided
    if transformation_graph is None:
        transformation_graph = build_transformation_graph()

    report = SolverAccessibilityReport(problem_type=problem_type)

    # 1. Find direct solvers
    direct_solvers = discover_solvers_for_problem(problem_type)
    for name, solver_class in direct_solvers.items():
        report.direct_solvers.append(
            SolverOption(
                problem_type=problem_type,
                solver_class=solver_class,
                solver_name=name,
                is_direct=True,
            )
        )

    # 2. Find solvers via transformations
    reachable = transformation_graph.get_reachable_problems(problem_type)

    for target_problem in reachable:
        if target_problem == problem_type:
            continue  # Skip self

        # Find path to this problem
        path = transformation_graph.find_path(
            problem_type, target_problem, WeightingStrategy.PREFER_EXACT
        )

        if path is None:
            continue

        # Find solvers for target problem
        target_solvers = discover_solvers_for_problem(target_problem)

        for name, solver_class in target_solvers.items():
            report.transformation_solvers.append(
                SolverOption(
                    problem_type=target_problem,
                    solver_class=solver_class,
                    solver_name=name,
                    path=path,
                    is_direct=False,
                )
            )

    return report


def compare_solver_options(
    problem_type: str,
    max_path_length: int = 3,
) -> None:
    """Compare solver options for a problem (with transformation graph).

    Args:
        problem_type: Problem class name
        max_path_length: Maximum transformation path length

    """
    print("=" * 80)
    print(f"Discovering Solvers for {problem_type}")
    print("=" * 80)

    # Build graph
    print("\nBuilding transformation graph...")
    graph = build_transformation_graph()
    graph.print_summary()

    # Build report
    print(f"\nAnalyzing solver accessibility...")
    report = build_solver_accessibility_report(problem_type, graph, max_path_length)

    # Print report
    report.print_report()

    # Additional analysis
    print("\n" + "=" * 80)
    print("Transformation Paths to Solvers")
    print("=" * 80)

    if report.transformation_solvers:
        # Show unique paths
        unique_paths = {}
        for solver in report.transformation_solvers:
            path_key = tuple(solver.path.problem_sequence)
            if path_key not in unique_paths:
                unique_paths[path_key] = solver.path

        print(f"\nUnique transformation paths: {len(unique_paths)}")
        for i, (path_key, path) in enumerate(sorted(unique_paths.items()), 1):
            exact_str = (
                "exact" if path.is_exact else f"lossy (weight: {path.total_weight:.2f})"
            )
            print(f"\n  Path {i} ({exact_str}):")
            for j, (src, edge) in enumerate(
                zip(path.problem_sequence[:-1], path.transformations)
            ):
                print(f"    {j + 1}. {edge.transformation_class.__name__}")
                print(f"       {edge.source_problem} → {edge.target_problem}")
                if not edge.forward_exact:
                    print(f"       ⚠ Forward lossy (impact: {edge.max_impact.value})")

    else:
        print("\nNo solvers accessible via transformations")

    # Summary statistics
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total solvers: {report.total_solvers}")
    print(f"  - Direct: {len(report.direct_solvers)}")
    print(f"  - Via transformations: {len(report.transformation_solvers)}")

    if report.transformation_solvers:
        exact_transformations = sum(
            1 for s in report.transformation_solvers if s.path.is_exact
        )
        print(f"    • Via exact transformations: {exact_transformations}")
        print(
            f"    • Via lossy transformations: {len(report.transformation_solvers) - exact_transformations}"
        )


def find_best_transformation_path(
    source: str,
    target: str,
    transformation_graph: Optional[TransformationGraph] = None,
) -> Optional[TransformationPath]:
    """Find best transformation path between problems.

    "Best" = prefer exact, then shortest, then lowest weight.

    Args:
        source: Source problem type
        target: Target problem type
        transformation_graph: Optional pre-built graph

    Returns:
        Best TransformationPath or None

    """
    if transformation_graph is None:
        transformation_graph = build_transformation_graph()

    # Find all paths
    all_paths = transformation_graph.find_all_paths(source, target, max_length=5)

    if not all_paths:
        return None

    # Sort by: exact first, then by weight
    all_paths.sort(
        key=lambda p: (not p.is_exact, p.total_weight, len(p.problem_sequence))
    )

    return all_paths[0]
