from copy import deepcopy
from typing import Dict, List, Type, Union

import networkx as nx
import numpy as np

from discrete_optimization.generic_tools.do_problem import (
    EncodingRegister,
    ModeOptim,
    ObjectiveDoc,
    ObjectiveHandling,
    ObjectiveRegister,
    Problem,
    Solution,
    TypeAttribute,
    TypeObjective,
)
from discrete_optimization.generic_tools.graph_api import Graph


class MisSolution(Solution):
    def __init__(self, problem: "MisProblem", chosen: Union[List, np.ndarray]):
        self.problem = problem
        self.chosen = chosen

    def copy(self) -> Solution:
        return MisSolution(problem=self.problem, chosen=deepcopy(self.chosen))

    def lazy_copy(self) -> Solution:
        return MisSolution(problem=self.problem, chosen=self.chosen)

    def change_problem(self, new_problem: "Problem") -> None:
        self.problem = new_problem


class MisProblem(Problem):
    def __init__(
        self, graph: Union[Graph, nx.Graph], attribute_aggregate: str = "size"
    ):
        self.graph = graph
        if isinstance(graph, Graph):
            self.nodes = self.graph.get_nodes()
            self.edges = self.graph.get_edges()
            self.graph_nx = self.graph.graph_nx
        else:
            self.nodes = list(self.graph.nodes())
            self.edges = list(self.graph.edges())
            self.graph_nx = self.graph

        self.number_nodes = len(self.nodes)
        self.attribute_aggregate = attribute_aggregate
        self.nodes_to_index = {self.nodes[i]: i for i in range(len(self.nodes))}
        self.index_to_nodes = {i: self.nodes[i] for i in range(len(self.nodes))}
        if self.attribute_aggregate == "size":
            self.attr_list = [1 for _ in self.index_to_nodes]
            self.func = sum
        else:
            if isinstance(self.graph, Graph):
                self.attr_list = [
                    self.graph.get_attr_node(self.nodes[i], self.attribute_aggregate)
                    for i in range(self.number_nodes)
                ]
            else:
                value = nx.get_node_attributes(self.graph, "value")
                attr_list = []
                for node in self.graph_nx.nodes:
                    attr_list.append(value[node])
                self.attr_list = attr_list
            self.func = sum

    def evaluate(self, variable: MisSolution) -> Dict[str, float]:
        print(variable)
        return {
            "value": self.func(variable.chosen),
            "penalty": self.compute_violation(variable),
        }

    def satisfy(self, variable: MisSolution) -> bool:
        return self.compute_violation(variable) == 0

    def compute_violation(self, variable: MisSolution) -> int:
        v = 0
        for e in self.edges:
            if (
                variable.chosen[self.nodes_to_index[e[0]]]
                == variable.chosen[self.nodes_to_index[e[1]]]
                == 1
            ):
                v += 1
        return v

    def get_attribute_register(self) -> EncodingRegister:
        return EncodingRegister(
            dict_attribute_to_type={
                "chosen": {
                    "name": "chosen",
                    "type": [TypeAttribute.LIST_BOOLEAN],
                    "n": len(self.nodes),
                }
            }
        )

    def get_solution_type(self) -> Type[Solution]:
        return MisSolution

    def get_objective_register(self) -> ObjectiveRegister:
        return ObjectiveRegister(
            objective_sense=ModeOptim.MAXIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc={
                "value": ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=1),
                "penalty": ObjectiveDoc(
                    type=TypeObjective.OBJECTIVE, default_weight=-100
                ),
            },
        )
