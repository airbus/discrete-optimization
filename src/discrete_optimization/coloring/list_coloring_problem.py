#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""List Coloring Problem - a variant of graph coloring with restricted color sets per node.

In list coloring, each node has a list of allowed colors (a subset of all colors).
The goal is to assign colors to nodes such that:
1. Adjacent nodes have different colors (standard coloring constraint)
2. Each node is assigned a color from its allowed list
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Optional

from discrete_optimization.coloring.problem import (
    ColoringConstraints,
    ColoringProblem,
    ColoringSolution,
)
from discrete_optimization.generic_tools.graph_api import Graph

Node = Hashable
Color = int


class ListColoringProblem(ColoringProblem):
    """List Coloring Problem with restricted color sets per node.

    This is a generalization of graph coloring where each node has a list
    of allowed colors. This naturally models assignment problems where:
    - Nodes represent tasks/jobs
    - Colors represent resources/workers/teams
    - Allowed colors represent which resources are compatible with each task

    Attributes:
        graph (Graph): Graph representing adjacency constraints
        allowed_colors (dict[Node, set[Color]]): For each node, the set of allowed colors
        number_of_nodes (int): Number of nodes in the graph
        subset_nodes (set[Hashable]): Subset of nodes to optimize
        nodes_name (list[Hashable]): List of node IDs
        constraints_coloring (ColoringConstraints): Fixed color constraints

    """

    def __init__(
        self,
        graph: Graph,
        allowed_colors: dict[Node, set[Color]],
        subset_nodes: Optional[set[Hashable]] = None,
        constraints_coloring: Optional[ColoringConstraints] = None,
    ):
        """Initialize List Coloring Problem.

        Args:
            graph: Graph with nodes and edges (adjacency constraints)
            allowed_colors: For each node, the set of allowed colors
            subset_nodes: Subset of nodes to consider in optimization (optional)
            constraints_coloring: Fixed color constraints (optional)

        """
        super().__init__(
            graph=graph,
            subset_nodes=subset_nodes,
            constraints_coloring=constraints_coloring,
        )
        self.allowed_colors = allowed_colors

        # Validate that all nodes have allowed colors defined
        for node in self.nodes_name:
            if node not in self.allowed_colors:
                # Default: all colors allowed if not specified
                self.allowed_colors[node] = set(range(self.number_of_nodes))

    def satisfy(self, variable: ColoringSolution) -> bool:  # type: ignore
        """Check if solution satisfies all constraints.

        Checks:
        1. Standard coloring constraint (adjacent nodes different colors)
        2. List coloring constraint (each node uses allowed color)

        Args:
            variable: Coloring solution to check

        Returns:
            True if solution is feasible

        """
        # First check standard coloring constraints
        if not super().satisfy(variable):
            return False

        # Check list coloring constraints (allowed colors)
        if variable.colors is None:
            return False

        for node in self.nodes_name:
            node_idx = self.index_nodes_name[node]
            color = variable.colors[node_idx]

            # Check if color is in allowed set
            if color not in self.allowed_colors[node]:
                return False

        return True

    def evaluate(self, variable: ColoringSolution) -> dict[str, float]:  # type: ignore
        """Evaluate solution with list coloring violations.

        Returns:
            Dictionary with:
            - nb_colors: Number of distinct colors used (in subset_nodes)
            - nb_violations: Number of edge violations (adjacent same color)
            - nb_list_violations: Number of nodes using non-allowed colors

        """
        # Get base evaluation (nb_colors, nb_violations)
        result = super().evaluate(variable)

        # Add list coloring violations
        nb_list_violations = 0
        if variable.colors is not None:
            for node in self.nodes_name:
                node_idx = self.index_nodes_name[node]
                color = variable.colors[node_idx]

                if color not in self.allowed_colors[node]:
                    nb_list_violations += 1

        result["nb_list_violations"] = nb_list_violations

        return result

    def get_max_color_index(self) -> int:
        """Get the maximum color index needed.

        Returns:
            Maximum color value across all allowed color sets

        """
        if not self.allowed_colors:
            return self.number_of_nodes - 1

        max_color = 0
        for color_set in self.allowed_colors.values():
            if color_set:
                max_color = max(max_color, max(color_set))

        return max_color

    def get_nb_colors_available(self, node: Node) -> int:
        """Get number of colors available for a given node.

        Args:
            node: Node to query

        Returns:
            Number of allowed colors for this node

        """
        return len(self.allowed_colors.get(node, set()))
