#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tools.graph_api import Graph, nx


def test_graph_api():
    nodes = [(0, {"name": 0}), (1, {"name": 1})]
    edges = [(0, 1, {"weight": 1.1}), (1, 0, {"weight": 2})]
    graph = Graph(nodes, edges, False)
    graph_nx = graph.to_networkx()
    assert graph.get_attr_edge(0, 1, "weight") == 1.1
    assert graph.get_attr_edge(1, 0, "weight") == 2
    assert graph.get_attr_edge(0, 0, "weight") is None
    assert graph_nx.size() == 2
    assert nx.number_of_nodes(graph_nx) == 2
    assert nx.number_of_edges(graph_nx) == 2
    assert graph_nx[0][1]["weight"] == 1.1
    assert graph_nx[1][0]["weight"] == 2
