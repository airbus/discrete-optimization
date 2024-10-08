#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from collections.abc import Iterator
from typing import Any

import didppy as dp
import networkx as nx

from discrete_optimization.coloring.coloring_model import (
    ColoringProblem,
    ColoringSolution,
)
from discrete_optimization.coloring.solvers.coloring_solver import SolverColoring
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.dyn_prog_tools import DidSolver


def bfs_iterator(G: nx.Graph, source: Any) -> Iterator[Any]:
    visited = set()
    queue = [source]
    visited.add(source)

    while queue:
        node = queue.pop(0)
        yield node

        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)


class DidColoringSolver(DidSolver, SolverColoring):
    def init_model(self, **kwargs: Any) -> None:
        graph = self.problem.graph.graph_nx
        nb_colors = kwargs.get("nb_colors", 50)
        degrees = {x[0]: x[1] for x in nx.degree(graph)}
        most_neighbor = max(degrees, key=lambda x: degrees[x])
        nodes_order = list(bfs_iterator(graph, source=most_neighbor))
        while len(nodes_order) < self.problem.number_of_nodes:
            n = next(x for x in self.problem.nodes_name if x not in nodes_order)
            nodes_order += list(bfs_iterator(graph, source=n))
        nodes_order = sorted(degrees, key=lambda x: degrees[x], reverse=True)
        nodes_index = [self.problem.index_nodes_name[n] for n in nodes_order]
        index_problem_to_model = {nodes_index[i]: i for i in range(len(nodes_index))}
        self.nodes_index = nodes_index
        model = dp.Model()
        colors = [model.add_int_var(target=0)] + [
            model.add_int_var(target=5 * i)
            for i in range(1, self.problem.number_of_nodes)
        ]
        node = model.add_object_type(number=self.problem.number_of_nodes)
        cur_color = model.add_int_var(target=0)
        uncolored = model.add_set_var(
            object_type=node, target=range(1, self.problem.number_of_nodes)
        )
        model.add_base_case([uncolored.is_empty()])
        for i in range(1, self.problem.number_of_nodes):
            cur_node = nodes_index[i]
            neigh = self.problem.graph.get_neighbors(self.problem.nodes_name[cur_node])
            neighs = [
                index_problem_to_model[self.problem.index_nodes_name[n]] for n in neigh
            ]
            for c in range(nb_colors):
                color = dp.Transition(
                    name=f"{i,c}",
                    cost=dp.max(c, cur_color) - cur_color + dp.IntExpr.state_cost(),
                    effects=[
                        (uncolored, uncolored.remove(i)),
                        (cur_color, dp.max(c, cur_color)),
                        (colors[i], c),
                    ],
                    preconditions=[
                        c <= cur_color + 1,
                        uncolored.contains(i),
                        ~uncolored.contains(i - 1),
                    ]
                    + [colors[n] != c for n in neighs],
                )
                model.add_transition(color)
        self.model = model

    def retrieve_solution(self, sol: dp.Solution) -> Solution:
        def extract_numbers(s):
            s = s.strip("()")
            return tuple(int(num) for num in s.split(","))

        colors = [None for _ in range(self.problem.number_of_nodes)]
        colors[self.nodes_index[0]] = 0
        for t in sol.transitions:
            n, c = extract_numbers(t.name)
            print(t.name)
            # print(self.nodes_index[n])
            # print(n, c)
            colors[self.nodes_index[n]] = c
        return ColoringSolution(problem=self.problem, colors=colors)
