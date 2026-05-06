#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Utility functions shared across multibatching solvers."""

import logging
from collections import defaultdict
from typing import Dict, Set

from discrete_optimization.multibatching.problem import MultibatchingProblem

logger = logging.getLogger(__name__)


def precompute_valid_links(
    problem: MultibatchingProblem, tolerance: float = 0.1
) -> Dict[str, Set[int]]:
    """
    Robust Heuristic: Identifies which transport links are relevant for each product.

    This function builds a specific graph G_p for each product P, containing only
    the TransportLinks compatible with P (i.e. link.transport_type in p.valid_transports).

    A link (u, v) is relevant for product P if there exists ANY pair of
    (Source s, Sink d) for P such that:
       dist_p(s, u) + link_len(u, v) + dist_p(v, d) <= (1 + tolerance) * dist_p(s, d)

    This heuristic helps reduce the search space by eliminating links that are
    unlikely to be part of optimal solutions (e.g., links that create long detours).

    Args:
        problem: The multibatching problem instance
        tolerance: Tolerance for path length relaxation (default: 0.1 = 10% longer than optimal)
                  - 0.0 = only shortest paths allowed
                  - 0.2 = paths up to 20% longer than optimal allowed
                  - Higher values = more links considered valid

    Returns:
        Dictionary mapping product IDs to sets of valid transport link indices.
        Format: {product.id: {link_index_1, link_index_2, ...}}

    Example:
        >>> valid_links = precompute_valid_links(problem, tolerance=0.2)
        >>> # Check if link 5 is valid for product with id "P1"
        >>> if 5 in valid_links.get("P1", set()):
        >>>     print("Link 5 can be used for product P1")
    """
    import networkx as nx

    valid_links_per_product = defaultdict(set)

    for p in problem.products:
        # 1. Identify Producers and Consumers for this specific product
        sources = [l.id for l in problem.locations if l.net_supply.get(p, 0) > 1e-6]
        sinks = [l.id for l in problem.locations if l.net_supply.get(p, 0) < -1e-6]

        if not sources or not sinks:
            logger.debug(
                f"Product {p.id}: No sources or sinks found, skipping shortest path analysis"
            )
            continue

        # 2. Build the Product-Specific Graph G_p
        # Only include links where the transport type is allowed for this product
        G_p = nx.DiGraph()

        # We also map (u, v) -> list of (link_index, weight)
        # because there might be multiple transport modes between u and v (e.g. Truck vs Train)
        # and we need to check them individually.
        edges_to_indices = defaultdict(list)

        for i, tl in enumerate(problem.transport_links):
            # KEY CHECK: Is this transport type valid for this product?
            # Also check capacity constraints if relevant (size <= capacity)
            if (
                tl.transport_type in p.valid_transports
                and p.size <= tl.transport_type.capacity
            ):
                u, v = tl.location_l1.id, tl.location_l2.id
                w = 1  # Use unit distance for path counting (could use tl.distance for weighted)

                # For NetworkX, we usually want the *best* weight between u and v
                # if we just want shortest path distance.
                # But we must store all parallel edges to evaluate them later.
                edges_to_indices[(u, v)].append((i, w))

                # Update graph with the shortest edge between u and v found so far
                if G_p.has_edge(u, v):
                    if w < G_p[u][v]["weight"]:
                        G_p[u][v]["weight"] = w
                else:
                    G_p.add_edge(u, v, weight=w)

        # 3. Compute All-Pairs Shortest Paths on G_p
        try:
            # This gives the shortest path distance matrix for valid moves only
            path_lengths = dict(nx.all_pairs_dijkstra_path_length(G_p, weight="weight"))
        except nx.NetworkXNoPath:
            logger.debug(f"Product {p.id}: Disconnected graph, skipping")
            continue  # Disconnected graph for this product

        # 4. Filter Links
        # We iterate over all physically existing links again, but only check valid ones
        for (u, v), link_list in edges_to_indices.items():
            # Pre-check: Can any source reach u?
            valid_sources = [
                s for s in sources if s in path_lengths and u in path_lengths[s]
            ]
            if not valid_sources:
                continue

            # Pre-check: Can v reach any sink?
            if v not in path_lengths:
                continue
            valid_sinks = [d for d in sinks if d in path_lengths[v]]
            if not valid_sinks:
                continue

            # Check each specific transport mode between u and v
            for index_tl, w in link_list:
                is_useful = False

                for s in valid_sources:
                    dist_s_u = path_lengths[s][u]

                    for d in valid_sinks:
                        if d not in path_lengths[s]:
                            continue  # Should not happen if graph is connected, but safety check

                        dist_s_d_opt = path_lengths[s][d]
                        dist_v_d = path_lengths[v][d]

                        # Check Path Quality
                        path_via_link = dist_s_u + w + dist_v_d

                        # Relaxed triangle inequality
                        if path_via_link <= dist_s_d_opt * (1.0 + tolerance) + 1e-5:
                            is_useful = True
                            break

                    if is_useful:
                        break

                if is_useful:
                    valid_links_per_product[p.id].add(index_tl)

    # Log statistics
    for p in problem.products:
        total_compatible = sum(
            1
            for tl in problem.transport_links
            if tl.transport_type in p.valid_transports
            and p.size <= tl.transport_type.capacity
        )
        valid_count = len(valid_links_per_product.get(p.id, set()))
        logger.info(
            f"Product {p.id}: {valid_count}/{total_compatible} compatible links "
            f"pass shortest path heuristic (tolerance={tolerance})"
        )

    return valid_links_per_product
