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
    CategoricalHyperparameter,
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


sol = """1   28   29   40   43   50   71   74   81  120  123  134  137  176  183
         186  207  214  217  228  229  256  260  261  288  303  310  313  334  340
         341  355  383  396  397  403  418  446  449  476  477  488  491  498  515
         543  558  564  565  588  589  595  610  638  648  651  658  673  700  701
         728  731  743  746  753  775  778  785  824  827  848  855  858  870  873
         911  918  921  932  933  960  963  991 1006 1012 1013 1026 1054 1068 1069
        1075 1096 1099 1106 1121 1148 1149 1159 1162 1169 1208 1211 1232 1239 1242
        1254 1257 1286 1289 1328 1335 1338 1359 1366 1369 1380 1381 1408 1422 1428
        1429 1443 1471 1474 1502 1516 1517 1523 1540 1541 1568 1583 1590 1593 1614
        1620 1621 1635 1663 1676 1677 1683 1698 1726 1729 1756 1757 1768 1771 1778
        1800 1803 1810 1825 1852 1853 1880 1883 1895 1898 1905 1936 1943 1946 1958
        1961 1988 1989 2016 2031 2038 2041"""
order = tuple(int(num) for num in re.findall(r"\d+", sol))


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
                    print(node)
                    solution.chosen[node] = 1
                except:
                    solution.chosen[self.nodes_convention[index]] = 1
                    index += 1
        return solution
