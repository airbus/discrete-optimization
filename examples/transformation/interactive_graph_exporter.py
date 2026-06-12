#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""
Export transformation graph to JSON for interactive visualization.

This script builds the transformation graph and exports it to JSON format
that can be used by the interactive HTML visualization.
"""

import json
from pathlib import Path

from discrete_optimization.generic_tools.transformation.solver_discovery import (
    discover_solvers_for_problem,
)
from discrete_optimization.generic_tools.transformation.transformation_graph import (
    build_transformation_graph,
)


def export_transformation_graph_to_json(output_file: str = "transformation_graph.json"):
    """Export transformation graph to JSON file.

    Args:
        output_file: Output JSON file path

    """
    print("Building transformation graph...")
    graph = build_transformation_graph()

    print(
        f"Found {graph.graph.number_of_nodes()} nodes and {graph.graph.number_of_edges()} edges"
    )

    # Build JSON structure
    data = {
        "nodes": [],
        "edges": [],
        "metadata": {
            "total_nodes": graph.graph.number_of_nodes(),
            "total_edges": graph.graph.number_of_edges(),
            "exact_transformations": 0,
            "lossy_transformations": 0,
        },
    }

    # Export nodes (problems)
    print("Discovering solvers for each problem...")
    for node in sorted(graph.graph.nodes()):
        # Get solvers for this problem
        solvers = discover_solvers_for_problem(node)
        solver_names = list(solvers.keys())

        # Get reachability info
        reachable = graph.get_reachable_problems(node)
        reachable_count = len(reachable) - 1  # Exclude self

        node_data = {
            "id": node,
            "label": node,
            "solvers": solver_names,
            "solver_count": len(solver_names),
            "reachable_count": reachable_count,
            "reachable_problems": sorted(list(reachable - {node})),
        }
        data["nodes"].append(node_data)

    # Export edges (transformations)
    print("Exporting transformations...")
    for (source, target), edge in sorted(graph.transformations.items()):
        # Get metadata
        metadata = edge.transformation_instance.get_forward_metadata()

        # Extract loss information
        losses = []
        for loss in metadata.losses:
            losses.append(
                {
                    "name": loss.name,
                    "type": loss.loss_type.value,
                    "description": loss.description,
                    "reason": loss.reason,
                    "impact": loss.impact.value,
                    "impact_severity": loss.impact.severity(),
                    "workaround": loss.workaround,
                }
            )

        # Determine edge type
        is_exact = edge.forward_exact and edge.backward_exact
        if is_exact:
            data["metadata"]["exact_transformations"] += 1
        else:
            data["metadata"]["lossy_transformations"] += 1

        edge_data = {
            "source": source,
            "target": target,
            "transformation_class": edge.transformation_class.__name__,
            "forward_exact": edge.forward_exact,
            "backward_exact": edge.backward_exact,
            "is_exact": is_exact,
            "max_impact": edge.max_impact.value,
            "max_impact_severity": edge.max_impact.severity(),
            "num_losses": edge.num_losses,
            "losses": losses,
            "completeness": metadata.completeness.value,
            "assumptions": metadata.assumptions,
            "use_cases": metadata.use_cases,
            "warnings": metadata.warnings,
            "gains": metadata.gains,
        }
        data["edges"].append(edge_data)

    # Add connected components info
    components = graph.get_connected_components()
    data["metadata"]["connected_components"] = len(components)
    data["metadata"]["components"] = [sorted(list(comp)) for comp in components]

    # Write to file
    output_path = Path(output_file)
    print(f"Writing to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✓ Successfully exported transformation graph to {output_path}")
    print(f"  - Nodes: {len(data['nodes'])}")
    print(f"  - Edges: {len(data['edges'])}")
    print(f"  - Exact transformations: {data['metadata']['exact_transformations']}")
    print(f"  - Lossy transformations: {data['metadata']['lossy_transformations']}")

    # Also export a summary for quick reference
    summary = {
        "problems": [node["id"] for node in data["nodes"]],
        "transformations": [
            {
                "from": edge["source"],
                "to": edge["target"],
                "name": edge["transformation_class"],
                "exact": edge["is_exact"],
            }
            for edge in data["edges"]
        ],
    }

    summary_path = output_path.with_stem(output_path.stem + "_summary")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Also wrote summary to {summary_path}")

    return data


if __name__ == "__main__":
    export_transformation_graph_to_json("transformation_graph.json")
