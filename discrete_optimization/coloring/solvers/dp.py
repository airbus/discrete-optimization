#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import re
from collections.abc import Iterator
from enum import Enum
from typing import Any

import didppy as dp
import networkx as nx

from discrete_optimization.coloring.problem import (
    ColoringSolution,
    transform_color_values_to_value_precede_on_other_node_order,
)
from discrete_optimization.coloring.solvers import ColoringSolver
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.dyn_prog_tools import DpSolver
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    EnumHyperparameter,
)


class DpColoringModeling(Enum):
    COLOR_TRANSITION = 0
    COLOR_NODE_TRANSITION = 1


def bfs_iterator(g: nx.Graph, source: Any) -> Iterator[Any]:
    visited = set()
    queue = [source]
    visited.add(source)

    while queue:
        node = queue.pop(0)
        yield node

        for neighbor in g.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)


class DpColoringSolver(DpSolver, ColoringSolver, WarmstartMixin):
    hyperparameters = DpSolver.hyperparameters + [
        EnumHyperparameter(
            name="modeling",
            enum=DpColoringModeling,
            default=DpColoringModeling.COLOR_TRANSITION,
        ),
        CategoricalHyperparameter(
            name="dual_bound", choices=[True, False], default=True
        ),
    ]
    transitions: dict
    nodes_reordering: list
    modeling: DpColoringModeling

    def init_model(self, **kwargs):
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        modeling: DpColoringModeling = kwargs["modeling"]
        if modeling == DpColoringModeling.COLOR_TRANSITION:
            self.init_model_color(**kwargs)
        if modeling == DpColoringModeling.COLOR_NODE_TRANSITION:
            self.init_model_color_and_node(**kwargs)
        self.modeling = modeling

    def init_model_color_and_node(self, **kwargs: Any) -> None:
        kwargs = self.complete_with_default_hyperparameters(kwargs)

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
        self.nodes_reordering = nodes_index
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
        self.transitions = {}
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
                self.transitions[(i, c)] = color
        self.model = model

    def init_model_color(self, **kwargs: Any) -> None:
        graph = self.problem.graph.graph_nx
        nb_colors = kwargs.get("nb_colors", 50)
        nb_nodes = self.problem.number_of_nodes

        degrees = {x[0]: x[1] for x in nx.degree(graph)}
        most_neighbor = max(degrees, key=lambda x: degrees[x])
        nodes_order = list(bfs_iterator(graph, source=most_neighbor))
        while len(nodes_order) < self.problem.number_of_nodes:
            n = next(x for x in self.problem.nodes_name if x not in nodes_order)
            nodes_order += list(bfs_iterator(graph, source=n))
        nodes_order = sorted(degrees, key=lambda x: degrees[x], reverse=True)
        nodes_index = [self.problem.index_nodes_name[n] for n in nodes_order]
        index_problem_to_model = {nodes_index[i]: i for i in range(len(nodes_index))}
        self.nodes_reordering = nodes_index
        model = dp.Model()
        node_type = model.add_object_type(number=nb_nodes)
        node_allocated_per_color = [
            model.add_set_var(object_type=node_type, target=set())
            for _ in range(nb_colors)
        ]
        current_node = model.add_element_var(object_type=node_type, target=0)
        neighbors = [
            {
                index_problem_to_model[self.problem.index_nodes_name[n]]
                for n in self.problem.graph.get_neighbors(
                    self.problem.index_to_nodes_name[nodes_index[i]]
                )
            }
            for i in range(nb_nodes)
        ]
        neighbors_tab = model.add_set_table(neighbors, object_type=node_type)
        nb_color_used = model.add_int_resource_var(target=0)
        self.transitions = {}
        for c in range(nb_colors):
            if c == 0:
                alloc = dp.Transition(
                    name=f"alloc_{c}",
                    cost=1 + dp.IntExpr.state_cost(),
                    effects=[
                        (
                            node_allocated_per_color[c],
                            node_allocated_per_color[c].add(current_node),
                        ),
                        (current_node, current_node + 1),
                        (nb_color_used, nb_color_used + 1),
                    ],
                    preconditions=[
                        current_node < nb_nodes,
                        node_allocated_per_color[c]
                        .intersection(neighbors_tab[current_node])
                        .is_empty(),
                        node_allocated_per_color[c].is_empty(),
                    ],
                )
                model.add_transition(alloc)
                self.transitions[("new_color", 0)] = alloc
            else:
                alloc = dp.Transition(
                    name=f"alloc_{c}",
                    cost=1 + dp.IntExpr.state_cost(),
                    effects=[
                        (
                            node_allocated_per_color[c],
                            node_allocated_per_color[c].add(current_node),
                        ),
                        (current_node, current_node + 1),
                        (nb_color_used, nb_color_used + 1),
                    ],
                    preconditions=[
                        current_node < nb_nodes,
                        node_allocated_per_color[c]
                        .intersection(neighbors_tab[current_node])
                        .is_empty(),
                        node_allocated_per_color[c].is_empty(),
                        ~node_allocated_per_color[c - 1].is_empty(),
                    ],
                )
                model.add_transition(alloc)
                self.transitions[("new_color", c)] = alloc
            alloc_no_new = dp.Transition(
                name=f"alloc_{c}_no_new",
                cost=dp.IntExpr.state_cost(),
                effects=[
                    (
                        node_allocated_per_color[c],
                        node_allocated_per_color[c].add(current_node),
                    ),
                    (current_node, current_node + 1),
                ],
                preconditions=[
                    current_node < nb_nodes,
                    node_allocated_per_color[c]
                    .intersection(neighbors_tab[current_node])
                    .is_empty(),
                    ~node_allocated_per_color[c].is_empty(),
                ],
            )
            model.add_transition(alloc_no_new)
            self.transitions[("not_new", c)] = alloc_no_new

        model.add_base_case([current_node == nb_nodes])
        if kwargs["dual_bound"]:
            model.add_dual_bound(0)
        self.model = model

    def retrieve_solution(self, sol: dp.Solution) -> Solution:
        if self.modeling == DpColoringModeling.COLOR_TRANSITION:
            return self.retrieve_solution_color(sol)
        if self.modeling == DpColoringModeling.COLOR_NODE_TRANSITION:
            return self.retrieve_solution_color_and_node(sol)

    def retrieve_solution_color_and_node(self, sol: dp.Solution) -> Solution:
        def extract_numbers(s):
            return tuple(int(num) for num in re.findall(r"\d+", s))

        colors = [None for _ in range(self.problem.number_of_nodes)]
        colors[self.nodes_reordering[0]] = 0
        for t in sol.transitions:
            n, c = extract_numbers(t.name)
            colors[self.nodes_reordering[n]] = c
        return ColoringSolution(problem=self.problem, colors=colors)

    def retrieve_solution_color(self, sol: dp.Solution) -> Solution:
        def extract_numbers(s):
            return tuple(int(num) for num in re.findall(r"\d+", s))

        colors = [None for _ in range(self.problem.number_of_nodes)]
        ind = 0
        for t in sol.transitions:
            c = extract_numbers(t.name)[0]
            colors[self.nodes_reordering[ind]] = c
            ind += 1
        return ColoringSolution(problem=self.problem, colors=colors)

    def set_warm_start(self, solution: ColoringSolution) -> None:
        nodes_reordering = self.nodes_reordering
        new_vector = transform_color_values_to_value_precede_on_other_node_order(
            solution.colors, nodes_ordering=nodes_reordering
        )
        initial_solution = []
        if self.modeling == DpColoringModeling.COLOR_TRANSITION:
            set_colors = set()
            for ind in range(len(nodes_reordering)):
                c = new_vector[nodes_reordering[ind]]
                if c not in set_colors:
                    initial_solution.append(self.transitions[("new_color", c)])
                    set_colors.add(c)
                else:
                    initial_solution.append(self.transitions[("not_new", c)])
            self.initial_solution = initial_solution
        if self.modeling == DpColoringModeling.COLOR_NODE_TRANSITION:
            for ind in range(1, len(nodes_reordering)):
                c = new_vector[nodes_reordering[ind]]
                initial_solution.append(self.transitions[(ind, c)])
            self.initial_solution = initial_solution
