#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation graph for discovering solver accessibility.

This module builds a graph of problem transformations and uses it to:
- Discover all transformations in the codebase
- Find paths between problem types
- Calculate transformation quality (based on losses)
- List all solvers accessible for a given problem
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.generic_tools.transformation.transformation_metadata import (
    LossImpact,
)


class WeightingStrategy(Enum):
    """Strategy for weighting transformation edges."""

    UNIFORM = "uniform"  # All edges weight 1 (minimize hops)
    BY_IMPACT = "by_impact"  # Weight by max loss impact
    BY_LOSS_COUNT = "by_loss_count"  # Weight by number of losses
    PREFER_EXACT = "prefer_exact"  # Heavily penalize lossy transformations


@dataclass
class TransformationEdge:
    """Edge in transformation graph representing a transformation."""

    source_problem: str  # Problem type name
    target_problem: str  # Problem type name
    transformation_class: type[ProblemTransformation]
    transformation_instance: ProblemTransformation
    forward_exact: bool
    backward_exact: bool
    max_impact: LossImpact
    num_losses: int

    def get_weight(
        self, strategy: WeightingStrategy = WeightingStrategy.UNIFORM
    ) -> float:
        """Get edge weight based on strategy.

        Args:
            strategy: Weighting strategy

        Returns:
            Edge weight (lower = better)

        """
        if strategy == WeightingStrategy.UNIFORM:
            return 1.0

        elif strategy == WeightingStrategy.BY_IMPACT:
            # Weight by impact level (0-4)
            impact_weights = {
                LossImpact.NONE: 0.0,
                LossImpact.MINOR: 1.0,
                LossImpact.MODERATE: 2.0,
                LossImpact.MAJOR: 5.0,
                LossImpact.CRITICAL: 10.0,
            }
            return impact_weights.get(self.max_impact, 10.0)

        elif strategy == WeightingStrategy.BY_LOSS_COUNT:
            # Weight by number of losses
            return float(self.num_losses) if self.num_losses > 0 else 0.1

        elif strategy == WeightingStrategy.PREFER_EXACT:
            # Heavily penalize lossy transformations
            if self.forward_exact and self.backward_exact:
                return 0.1
            elif self.forward_exact or self.backward_exact:
                return 5.0  # One direction lossy
            else:
                return 20.0  # Both directions lossy

        else:
            return 1.0

    def __str__(self) -> str:
        """String representation."""
        exact_str = ""
        if self.forward_exact and self.backward_exact:
            exact_str = " (exact)"
        elif self.forward_exact:
            exact_str = " (forward exact)"
        elif self.backward_exact:
            exact_str = " (backward exact)"
        else:
            exact_str = f" (lossy, impact: {self.max_impact.value})"

        return f"{self.source_problem} → {self.target_problem}{exact_str}"


@dataclass
class TransformationPath:
    """Path through transformation graph."""

    problem_sequence: list[str]  # Sequence of problem types
    transformations: list[TransformationEdge]  # Edges used
    total_weight: float
    is_exact: bool  # True if all transformations are exact

    def __str__(self) -> str:
        """String representation."""
        path_str = " → ".join(self.problem_sequence)
        exact_str = (
            " (exact)" if self.is_exact else f" (weight: {self.total_weight:.2f})"
        )
        return f"{path_str}{exact_str}"


class TransformationGraph:
    """Graph of problem transformations.

    Builds and analyzes a graph where:
    - Nodes = problem types (e.g., "BinPackProblem", "SalbpProblem")
    - Edges = transformations with metadata

    """

    def __init__(self):
        """Initialize empty transformation graph."""
        if not NETWORKX_AVAILABLE:
            raise ImportError(
                "NetworkX required for TransformationGraph. Install with: pip install networkx"
            )

        self.graph: nx.DiGraph = nx.DiGraph()
        self.transformations: dict[tuple[str, str], TransformationEdge] = {}

    def add_transformation(
        self, transformation_class: type[ProblemTransformation]
    ) -> None:
        """Add a transformation to the graph.

        Args:
            transformation_class: Transformation class to add

        """
        # Instantiate transformation
        try:
            transformation = transformation_class()
        except Exception as e:
            print(
                f"Warning: Could not instantiate {transformation_class.__name__}: {e}"
            )
            return

        # Extract problem type names from generic type parameters
        # This is a bit hacky but works for our use case
        source_name, target_name = self._extract_problem_types(transformation_class)

        if not source_name or not target_name:
            print(
                f"Warning: Could not extract types from {transformation_class.__name__}"
            )
            return

        # Get metadata (only forward - backward is deprecated)
        forward_metadata = transformation.get_forward_metadata()

        # Create edge
        edge = TransformationEdge(
            source_problem=source_name,
            target_problem=target_name,
            transformation_class=transformation_class,
            transformation_instance=transformation,
            forward_exact=forward_metadata.is_exact(),
            backward_exact=True,  # Solution mapping is always mechanical (deprecated concept)
            max_impact=forward_metadata.get_max_impact(),
            num_losses=len(forward_metadata.losses),
        )

        # Add to graph
        self.graph.add_node(source_name)
        self.graph.add_node(target_name)
        self.graph.add_edge(source_name, target_name, transformation=edge)
        self.transformations[(source_name, target_name)] = edge

    def _extract_problem_types(
        self, transformation_class: type[ProblemTransformation]
    ) -> tuple[Optional[str], Optional[str]]:
        """Extract source and target problem type names.

        Args:
            transformation_class: Transformation class

        Returns:
            Tuple of (source_problem_name, target_problem_name)

        """
        # Try to get from class name convention
        # e.g., BinpackToSalbpTransformation -> BinPackProblem, SalbpProblem
        class_name = transformation_class.__name__

        # Parse from name like "BinpackToSalbpTransformation"
        if "To" in class_name and "Transformation" in class_name:
            parts = class_name.replace("Transformation", "").split("To")
            if len(parts) == 2:
                source = parts[0].strip()
                target = parts[1].strip()

                # Convert to problem names (add "Problem" suffix)
                # Handle special cases
                source_problem = self._to_problem_name(source)
                target_problem = self._to_problem_name(target)

                return source_problem, target_problem

        return None, None

    def _to_problem_name(self, short_name: str) -> str:
        """Convert short name to problem class name.

        Args:
            short_name: Short name (e.g., "Binpack", "Salbp")

        Returns:
            Problem class name (e.g., "BinPackProblem", "SalbpProblem")

        """
        # Special cases
        special_cases = {
            "Binpack": "BinPackProblem",
            "Salbp": "SalbpProblem",
            "RcalbpL": "RCALBPLProblem",
            "Rcpsp": "RcpspProblem",
            "Multiskill": "MultiskillRcpspProblem",
            "Preemptive": "PreemptiveRcpspProblem",
            "Fjsp": "FJobShopProblem",
            "Jsp": "JobShopProblem",
            "Facility": "FacilityProblem",
            "Singlebatch": "SingleBatchProcessingProblem",
            "Ovensched": "OvenSchedulingProblem",
            "WorkforceAllocation": "TeamAllocationProblem",
            "WorkforceScheduling": "AllocSchedulingProblem",
            "Coloring": "ColoringProblem",
            "ListColoring": "ListColoringProblem",
            "Tsp": "TspProblem",
            "Vrp": "VrpProblem",
            "Vrptw": "VRPTWProblem",
            "Gpdp": "GpdpProblem",
            "Top": "TeamOrienteeringProblem",
        }

        if short_name in special_cases:
            return special_cases[short_name]

        # Default: add "Problem" suffix
        return f"{short_name}Problem"

    def find_path(
        self,
        source: str,
        target: str,
        strategy: WeightingStrategy = WeightingStrategy.UNIFORM,
    ) -> Optional[TransformationPath]:
        """Find shortest path between two problem types.

        Args:
            source: Source problem type name
            target: Target problem type name
            strategy: Weighting strategy for path finding

        Returns:
            TransformationPath if path exists, None otherwise

        """
        # Set edge weights based on strategy
        for (src, tgt), edge in self.transformations.items():
            self.graph[src][tgt]["weight"] = edge.get_weight(strategy)

        try:
            path = nx.shortest_path(self.graph, source, target, weight="weight")
            path_weight = nx.shortest_path_length(
                self.graph, source, target, weight="weight"
            )

            # Extract transformations along path
            transformations = []
            for i in range(len(path) - 1):
                edge = self.transformations[(path[i], path[i + 1])]
                transformations.append(edge)

            # Check if entire path is exact
            is_exact = all(
                t.forward_exact and t.backward_exact for t in transformations
            )

            return TransformationPath(
                problem_sequence=path,
                transformations=transformations,
                total_weight=path_weight,
                is_exact=is_exact,
            )

        except nx.NetworkXNoPath:
            return None

    def find_all_paths(
        self,
        source: str,
        target: str,
        max_length: int = 5,
    ) -> list[TransformationPath]:
        """Find all paths between two problem types.

        Args:
            source: Source problem type name
            target: Target problem type name
            max_length: Maximum path length

        Returns:
            List of TransformationPath objects

        """
        paths = []
        try:
            for path_nodes in nx.all_simple_paths(
                self.graph, source, target, cutoff=max_length
            ):
                # Extract transformations
                transformations = []
                total_weight = 0.0
                for i in range(len(path_nodes) - 1):
                    edge = self.transformations[(path_nodes[i], path_nodes[i + 1])]
                    transformations.append(edge)
                    total_weight += edge.get_weight(WeightingStrategy.BY_IMPACT)

                is_exact = all(
                    t.forward_exact and t.backward_exact for t in transformations
                )

                paths.append(
                    TransformationPath(
                        problem_sequence=path_nodes,
                        transformations=transformations,
                        total_weight=total_weight,
                        is_exact=is_exact,
                    )
                )

        except nx.NetworkXNoPath:
            pass

        return paths

    def get_reachable_problems(self, source: str) -> set[str]:
        """Get all problem types reachable from source.

        Args:
            source: Source problem type

        Returns:
            Set of reachable problem type names

        """
        try:
            return set(nx.descendants(self.graph, source)) | {source}
        except nx.NetworkXError:
            return {source}

    def get_connected_components(self) -> list[set[str]]:
        """Get weakly connected components.

        Returns:
            List of sets, each containing connected problem types

        """
        return list(nx.weakly_connected_components(self.graph))

    def print_summary(self) -> None:
        """Print summary of transformation graph."""
        print("Transformation Graph Summary")
        print("=" * 80)
        print(f"Nodes (problem types): {self.graph.number_of_nodes()}")
        print(f"Edges (transformations): {self.graph.number_of_edges()}")

        # Count exact vs lossy
        exact_count = sum(
            1
            for edge in self.transformations.values()
            if edge.forward_exact and edge.backward_exact
        )
        print(f"  - Exact transformations: {exact_count}")
        print(f"  - Lossy transformations: {len(self.transformations) - exact_count}")

        # Connected components
        components = self.get_connected_components()
        print(f"\nConnected components: {len(components)}")
        for i, component in enumerate(components, 1):
            print(f"  Component {i}: {', '.join(sorted(component))}")

    def visualize(self, output_file: Optional[str] = None) -> None:
        """Visualize transformation graph.

        Args:
            output_file: Optional output file path (requires matplotlib/graphviz)

        """
        try:
            import matplotlib.pyplot as plt

            pos = nx.spring_layout(self.graph, k=2, iterations=50)

            # Color nodes
            node_colors = []
            for node in self.graph.nodes():
                # Color by reachability
                reachable = len(self.get_reachable_problems(node))
                node_colors.append(reachable)

            # Draw
            nx.draw_networkx_nodes(
                self.graph, pos, node_color=node_colors, cmap="Blues", node_size=1000
            )
            nx.draw_networkx_labels(self.graph, pos, font_size=8)

            # Draw edges with colors based on exactness
            exact_edges = [
                (u, v)
                for u, v, data in self.graph.edges(data=True)
                if data["transformation"].forward_exact
                and data["transformation"].backward_exact
            ]
            lossy_edges = [
                (u, v)
                for u, v, data in self.graph.edges(data=True)
                if not (
                    data["transformation"].forward_exact
                    and data["transformation"].backward_exact
                )
            ]

            nx.draw_networkx_edges(
                self.graph, pos, edgelist=exact_edges, edge_color="green", width=2
            )
            nx.draw_networkx_edges(
                self.graph,
                pos,
                edgelist=lossy_edges,
                edge_color="red",
                width=1,
                style="dashed",
            )

            plt.title("Transformation Graph (green=exact, red=lossy)")
            plt.axis("off")

            if output_file:
                plt.savefig(output_file, dpi=150, bbox_inches="tight")
                print(f"Saved visualization to {output_file}")
            else:
                plt.show()

        except ImportError:
            print(
                "Matplotlib required for visualization. Install with: pip install matplotlib"
            )


def discover_transformations(
    base_module: str = "discrete_optimization",
) -> TransformationGraph:
    """Discover all transformations in codebase.

    Args:
        base_module: Base module to search (default: discrete_optimization)

    Returns:
        TransformationGraph with all discovered transformations

    """
    graph = TransformationGraph()

    # Import base module
    try:
        base = importlib.import_module(base_module)
    except ImportError as e:
        print(f"Could not import base module {base_module}: {e}")
        return graph

    # Walk through all submodules
    base_path = Path(base.__file__).parent

    # Helper function to recursively find all transformations directories
    def find_transformation_modules(root_path: Path, module_prefix: str) -> list[str]:
        """Find all transformation module paths recursively."""
        transformation_modules = []

        for item in root_path.iterdir():
            if (
                not item.is_dir()
                or item.name.startswith("_")
                or item.name == "__pycache__"
            ):
                continue

            # Check if this directory has a transformations subdir
            transformations_dir = item / "transformations"
            if transformations_dir.exists():
                module_path = f"{module_prefix}.{item.name}.transformations"
                transformation_modules.append(module_path)

            # Recursively search subdirectories (for cases like workforce/scheduling/transformations)
            sub_modules = find_transformation_modules(
                item, f"{module_prefix}.{item.name}"
            )
            transformation_modules.extend(sub_modules)

        return transformation_modules

    # Find all transformation modules (including nested ones)
    transformation_module_paths = find_transformation_modules(base_path, base_module)

    # Import and process each transformation module
    for module_path in transformation_module_paths:
        try:
            transformations_module = importlib.import_module(module_path)

            # Find all transformation classes
            for name, obj in inspect.getmembers(transformations_module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, ProblemTransformation)
                    and obj != ProblemTransformation
                    and not inspect.isabstract(obj)
                ):
                    graph.add_transformation(obj)

        except ImportError as e:
            # Skip modules that can't be imported (missing dependencies)
            pass
        except Exception as e:
            print(f"Error processing {module_path}: {e}")

    return graph


def build_transformation_graph() -> TransformationGraph:
    """Build transformation graph from codebase.

    Returns:
        TransformationGraph with all discovered transformations

    """
    return discover_transformations()
