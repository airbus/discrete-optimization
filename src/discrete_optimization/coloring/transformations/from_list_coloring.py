#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Transformation from ListColoringProblem to ColoringProblem using dummy nodes."""

from typing import Optional

from discrete_optimization.coloring.list_coloring_problem import ListColoringProblem
from discrete_optimization.coloring.problem import (
    ColoringConstraints,
    ColoringProblem,
    ColoringSolution,
)
from discrete_optimization.generic_tools.graph_api import Graph
from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)
from discrete_optimization.generic_tools.transformation.transformation_metadata import (
    TransformationMetadata,
    exact_transformation,
)


class ListColoringToColoringTransformation(
    ProblemTransformation[
        ListColoringProblem,
        ColoringSolution,
        ColoringProblem,
        ColoringSolution,
    ]
):
    """Transform ListColoringProblem to ColoringProblem using dummy nodes.

    Mapping:
    - Create K dummy nodes (one per color)
    - Dummy nodes form a complete clique (all connected)
    - Dummy node k is fixed to color k
    - Original node can use color k only if NOT connected to dummy node k
    - Use subset_nodes to focus optimization on original nodes only

    This encoding allows standard coloring solvers to solve list coloring problems.

    This transformation is EXACT in both directions.
    """

    def __init__(self):
        """Initialize transformation with mapping storage."""
        # Store mapping between dummy nodes and colors
        self.dummy_node_to_color = {}
        self.color_to_dummy_node = {}

    def get_forward_metadata(self) -> TransformationMetadata:
        """Metadata for forward problem transformation (ListColoring → Coloring).

        This direction is EXACT: list coloring can be perfectly encoded as standard coloring.
        """
        return exact_transformation(
            use_cases=[
                "Use standard ColoringProblem solvers for ListColoring",
                "Forbidden colors modeled as edges to dummy nodes",
                "Dummy nodes form a clique with fixed colors",
            ]
        )

    def transform_problem(self, source_problem: ListColoringProblem) -> ColoringProblem:
        """Transform ListColoringProblem to ColoringProblem with dummy nodes.

        Args:
            source_problem: ListColoringProblem instance

        Returns:
            Equivalent ColoringProblem with dummy color nodes

        """
        # Determine number of colors needed
        max_color = source_problem.get_max_color_index()
        nb_colors = max_color + 1

        # Create dummy color nodes and store mapping
        dummy_nodes = [f"__color_{k}__" for k in range(nb_colors)]

        # Store mapping for later use
        self.dummy_node_to_color = {dummy_nodes[k]: k for k in range(nb_colors)}
        self.color_to_dummy_node = {k: dummy_nodes[k] for k in range(nb_colors)}

        # Build new graph with original + dummy nodes
        nodes = list(source_problem.graph.nodes)  # Copy original nodes

        # Add dummy nodes
        for k, dummy in enumerate(dummy_nodes):
            nodes.append((dummy, {"dummy": True, "fixed_color": k}))

        # Build edges
        edges = list(source_problem.graph.edges)  # Copy original edges

        # Add clique edges between dummy nodes (all pairs)
        for i in range(len(dummy_nodes)):
            for j in range(i + 1, len(dummy_nodes)):
                edges.append((dummy_nodes[i], dummy_nodes[j], {}))
                edges.append((dummy_nodes[j], dummy_nodes[i], {}))

        # Add forbidden edges: node → dummy color node
        # If color k is NOT in allowed_colors[node], add edge node → dummy_k
        for node in source_problem.nodes_name:
            allowed = source_problem.allowed_colors.get(node, set())
            for k in range(nb_colors):
                if k not in allowed:
                    # Forbidden: add edge to prevent this color
                    edges.append((node, dummy_nodes[k], {}))
                    edges.append((dummy_nodes[k], node, {}))

        # Create graph
        graph = Graph(
            nodes=nodes,
            edges=edges,
            undirected=source_problem.graph.undirected,
            compute_predecessors=False,
        )

        # Create ColoringProblem with subset_nodes = original nodes only
        # This ensures optimization focuses on original nodes, not dummy nodes
        subset_nodes = set(source_problem.nodes_name)

        # Build color constraints for dummy nodes (fix their colors)
        color_constraint = {dummy_nodes[k]: k for k in range(nb_colors)}

        # Merge with existing constraints if any
        if source_problem.constraints_coloring is not None:
            color_constraint.update(
                source_problem.constraints_coloring.color_constraint
            )

        constraints_coloring = ColoringConstraints(color_constraint=color_constraint)

        return ColoringProblem(
            graph=graph,
            subset_nodes=subset_nodes,
            constraints_coloring=constraints_coloring,
        )

    def back_transform_solution(
        self, solution: ColoringSolution, source_problem: ListColoringProblem
    ) -> ColoringSolution:
        """Transform Coloring solution back to ListColoring solution.

        Args:
            solution: Coloring solution (with dummy nodes)
            source_problem: Original ListColoringProblem

        Returns:
            Equivalent ListColoring solution (without dummy nodes)

        """
        # Extract colors for original nodes only
        # Dummy node colors are ignored
        colors = []

        for node in source_problem.nodes_name:
            node_idx = solution.problem.index_nodes_name[node]
            color = solution.colors[node_idx]
            colors.append(color)

        return ColoringSolution(
            problem=source_problem,
            colors=colors,
            nb_color=None,  # Will be recomputed
            nb_violations=None,  # Will be recomputed
        )

    def forward_transform_solution(
        self, solution: ColoringSolution, target_problem: ColoringProblem
    ) -> Optional[ColoringSolution]:
        """Transform ListColoring solution to Coloring solution (for warmstart).

        Args:
            solution: ListColoring solution
            target_problem: Target ColoringProblem (with dummy nodes)

        Returns:
            Equivalent Coloring solution (with dummy node colors added)

        """
        # Build full color array: original nodes + dummy nodes
        # Dummy nodes get their fixed colors
        colors = [None] * target_problem.number_of_nodes

        # Fill original node colors from solution
        for i, node in enumerate(solution.problem.nodes_name):
            target_idx = target_problem.index_nodes_name[node]
            colors[target_idx] = solution.colors[i]

        # Fill dummy node colors (fixed) using stored mapping
        for node in target_problem.nodes_name:
            if node in self.dummy_node_to_color:
                # Dummy node: use stored mapping
                color_idx = self.dummy_node_to_color[node]
                target_idx = target_problem.index_nodes_name[node]
                colors[target_idx] = color_idx

        return ColoringSolution(
            problem=target_problem,
            colors=colors,
            nb_color=None,  # Will be recomputed
            nb_violations=None,  # Will be recomputed
        )
