#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import re
from enum import Enum
from typing import Any

import networkx as nx

from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.dyn_prog_tools import DidSolver, dp
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
)
from discrete_optimization.maximum_independent_set.mis_model import (
    MisProblem,
    MisSolution,
)
from discrete_optimization.maximum_independent_set.solvers.mis_solver import MisSolver


class DidModeling(Enum):
    ORDER = 0
    ANY_ORDER = 1


class DidMisSolver(DidSolver, MisSolver):
    problem: MisProblem
    hyperparameters = DidSolver.hyperparameters + [
        EnumHyperparameter(name="modeling", enum=DidModeling, default=DidModeling.ORDER)
    ]

    def init_model(self, **kwargs: Any) -> None:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        if kwargs["modeling"] == DidModeling.ORDER:
            self.dp_in_order_(**kwargs)
        if kwargs["modeling"] == DidModeling.ANY_ORDER:
            self.dp_in_any_order(**kwargs)

    def dp_in_order(self, **kwargs: Any):
        model = dp.Model(maximize=True)
        indexes = range(self.problem.number_nodes)

        nodes = model.add_object_type(number=self.problem.number_nodes)
        self.nodes_convention = [i for i in range(self.problem.number_nodes)]
        # closed = model.add_set_var(object_type=nodes, target=set())
        open = model.add_set_var(object_type=nodes, target=indexes)
        cur_node = model.add_element_var(object_type=nodes, target=0)
        model.add_base_case([open.is_empty()])
        to_remove = []
        for i in range(self.problem.number_nodes):
            to_remove.append(
                set(
                    [
                        self.problem.nodes_to_index[n]
                        for n in self.problem.graph.neighbors(
                            self.problem.index_to_nodes[i]
                        )
                    ]
                    + [i]
                )
            )
        nodes_to_remove = model.add_set_table(to_remove, object_type=nodes)
        take = dp.Transition(
            name=f"pick",
            cost=1 + dp.IntExpr.state_cost(),
            effects=[
                (open, open.difference(nodes_to_remove[cur_node])),
                (cur_node, cur_node + 1),
            ],
            preconditions=[open.contains(cur_node)],
        )
        model.add_transition(take)
        no_take = dp.Transition(
            name=f"no_pick",
            cost=dp.IntExpr.state_cost(),
            effects=[(open, open.remove(cur_node)), (cur_node, cur_node + 1)],
            preconditions=[
                # open.contains(cur_node)
            ],
        )
        model.add_transition(no_take)
        self.model = model

    def dp_in_order_(self, **kwargs: Any):
        model = dp.Model(maximize=True)
        indexes = range(self.problem.number_nodes)
        degrees = {x[0]: x[1] for x in nx.degree(self.problem.graph_nx)}
        degrees = nx.degree_centrality(self.problem.graph_nx)
        sorted_nodes = sorted(degrees, key=lambda x: degrees[x], reverse=True)
        # sorted_nodes = [self.problem.nodes_to_index[n] for n in sorted_nodes]
        # sorted_nodes = [self.problem.nodes_to_index[o] for o in order]
        # sorted_nodes += [i for i in range(self.problem.number_nodes)
        #                  if i not in sorted_nodes]
        sorted_nodes = [
            self.problem.nodes_to_index[p] for p in sorted(self.problem.nodes)
        ]
        self.nodes_convention = sorted_nodes
        self.original_node_to_node_convention = {
            sorted_nodes[i]: i for i in range(len(sorted_nodes))
        }
        nodes = model.add_object_type(number=self.problem.number_nodes)
        # closed = model.add_set_var(object_type=nodes, target=set())
        open = model.add_set_var(object_type=nodes, target=indexes)
        cur_node = model.add_element_var(object_type=nodes, target=0)
        model.add_base_case([open.is_empty()])
        to_remove = []
        for i in range(self.problem.number_nodes):
            n_ = self.nodes_convention[i]
            to_remove.append(
                set(
                    [
                        self.original_node_to_node_convention[
                            self.problem.nodes_to_index[n]
                        ]
                        for n in self.problem.graph.neighbors(
                            self.problem.index_to_nodes[n_]
                        )
                    ]
                    + [i]
                )
            )

        nodes_to_remove = model.add_set_table(to_remove, object_type=nodes)
        take = dp.Transition(
            name=f"pick",
            cost=1 + dp.IntExpr.state_cost(),
            effects=[
                (open, open.difference(nodes_to_remove[cur_node])),
                (cur_node, cur_node + 1),
            ],
            preconditions=[open.contains(cur_node)],
        )
        model.add_transition(take)
        no_take = dp.Transition(
            name=f"no_pick",
            cost=dp.IntExpr.state_cost(),
            effects=[(open, open.remove(cur_node)), (cur_node, cur_node + 1)],
            preconditions=[
                # open.contains(cur_node)
            ],
        )
        model.add_transition(no_take)
        self.model = model

    def dp_in_any_order(self, **kwargs: Any) -> None:
        model = dp.Model(maximize=True)
        indexes = range(self.problem.number_nodes)

        nodes = model.add_object_type(number=self.problem.number_nodes)
        # closed = model.add_set_var(object_type=nodes, target=set())
        open = model.add_set_var(object_type=nodes, target=indexes)
        model.add_base_case([open.is_empty()])
        to_remove = []
        for i in range(self.problem.number_nodes):
            to_remove.append(
                set(
                    [
                        self.problem.nodes_to_index[n]
                        for n in self.problem.graph.neighbors(
                            self.problem.index_to_nodes[i]
                        )
                    ]
                    + [i]
                )
            )
        nodes_to_remove = model.add_set_table(to_remove, object_type=nodes)
        for i in range(self.problem.number_nodes):
            take = dp.Transition(
                name=f"pick_{i}",
                cost=1 + dp.IntExpr.state_cost(),
                effects=[(open, open.difference(nodes_to_remove[i]))],
                preconditions=[open.contains(i)],
            )
            model.add_transition(take)
        self.model = model

    def retrieve_solution(self, sol: dp.Solution) -> Solution:
        def extract_ints(word):
            return tuple(int(num) for num in re.findall(r"\d+", word))

        solution = MisSolution(
            problem=self.problem, chosen=[0 for _ in range(self.problem.number_nodes)]
        )
        index = 0
        for t in sol.transitions:
            if "no_pick" in t.name:
                index += 1
                continue
            else:
                try:
                    node = extract_ints(t.name)[0]
                    solution.chosen[node] = 1
                except:
                    solution.chosen[self.nodes_convention[index]] = 1
                    index += 1
        return solution
