#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Hashable, KeysView, List, Optional, Set, Tuple

import networkx as nx


class Graph:
    def __init__(
        self,
        nodes: List[Tuple[Hashable, Dict[str, Any]]],
        edges: List[Tuple[Hashable, Hashable, Dict[str, Any]]],
        undirected: bool = True,
        compute_predecessors: bool = True,
    ):
        self.nodes = nodes
        self.edges = edges
        self.undirected = undirected
        self.neighbors_dict: Dict[Hashable, Set[Hashable]] = {}
        self.predecessors_dict: Dict[Hashable, Set[Hashable]] = {}
        self.edges_infos_dict: Dict[Tuple[Hashable, Hashable], Dict[str, Any]] = {}
        self.nodes_infos_dict: Dict[Hashable, Dict[str, Any]] = {}
        self.build_nodes_infos_dict()
        self.build_edges()
        self.nodes_name = list(self.nodes_infos_dict)
        self.graph_nx = self.to_networkx()
        self.full_predecessors: Optional[Dict[Hashable, Set[Hashable]]]
        self.full_successors: Optional[Dict[Hashable, Set[Hashable]]]
        if compute_predecessors:
            self.full_predecessors = self.ancestors_map()
            self.full_successors = self.descendants_map()
        else:
            self.full_predecessors = None
            self.full_successors = None

    def get_edges(self) -> KeysView[Tuple[Hashable, Hashable]]:
        return self.edges_infos_dict.keys()

    def get_nodes(self) -> List[Hashable]:
        return self.nodes_name

    def build_nodes_infos_dict(self) -> None:
        for n, d in self.nodes:
            self.nodes_infos_dict[n] = d

    def build_edges(self) -> None:
        for n1, n2, d in self.edges:
            self.edges_infos_dict[(n1, n2)] = d
            if n2 not in self.predecessors_dict:
                self.predecessors_dict[n2] = set()
            if n1 not in self.neighbors_dict:
                self.neighbors_dict[n1] = set()
            self.predecessors_dict[n2].add(n1)
            self.neighbors_dict[n1].add(n2)
            if self.undirected:
                if n1 not in self.predecessors_dict:
                    self.predecessors_dict[n1] = set()
                if n2 not in self.neighbors_dict:
                    self.neighbors_dict[n2] = set()
                self.predecessors_dict[n1].add(n2)
                self.neighbors_dict[n2].add(n1)
                self.edges_infos_dict[(n2, n1)] = d

    def get_neighbors(self, node: Hashable) -> Set[Hashable]:
        return self.neighbors_dict.get(node, set())

    def get_predecessors(self, node: Hashable) -> Set[Hashable]:
        return self.predecessors_dict.get(node, set())

    def get_attr_node(self, node: Hashable, attr: str) -> Any:
        return self.nodes_infos_dict.get(node, {}).get(attr, None)

    def get_attr_edge(self, node1: Hashable, node2: Hashable, attr: str) -> Any:
        return self.edges_infos_dict.get((node1, node2), {}).get(attr, None)

    def to_networkx(self) -> nx.DiGraph:
        graph_nx = nx.DiGraph() if not self.undirected else nx.Graph()
        graph_nx.add_nodes_from(self.nodes)
        graph_nx.add_edges_from(self.edges)
        return graph_nx

    def check_loop(self) -> Optional[List[Tuple[Hashable, Hashable, str]]]:
        try:
            cycles = nx.find_cycle(self.graph_nx, orientation="original")
        except:
            cycles = None
        return cycles

    def precedessors_nodes(self, n: Hashable) -> Set[Hashable]:
        return nx.algorithms.ancestors(self.graph_nx, n)

    def ancestors_map(self) -> Dict[Hashable, Set[Hashable]]:
        return {
            n: nx.algorithms.ancestors(self.graph_nx, n) for n in self.graph_nx.nodes()
        }

    def descendants_map(self) -> Dict[Hashable, Set[Hashable]]:
        return {
            n: nx.algorithms.descendants(self.graph_nx, n)
            for n in self.graph_nx.nodes()
        }

    def successors_map(self) -> Dict[Hashable, List[Hashable]]:
        return {n: list(nx.neighbors(self.graph_nx, n)) for n in self.graph_nx.nodes()}

    def predecessors_map(self) -> Dict[Hashable, List[Hashable]]:
        return {n: list(self.graph_nx.predecessors(n)) for n in self.graph_nx.nodes()}
