#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import re
from enum import Enum
from typing import Any

import networkx as nx

from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.dyn_prog_tools import DpSolver, dp
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
)
from discrete_optimization.maximum_independent_set.problem import (
    MisProblem,
    MisSolution,
)
from discrete_optimization.maximum_independent_set.solvers.mis_solver import MisSolver


class DpModeling(Enum):
    ORDER = 0
    ANY_ORDER = 1


class FixedOrderMethod(Enum):
    ORIGINAL_ORDER = 0
    DEGREE = 1


class DpMisSolver(DpSolver, MisSolver, WarmstartMixin):
    problem: MisProblem
    hyperparameters = DpSolver.hyperparameters + [
        EnumHyperparameter(name="modeling", enum=DpModeling, default=DpModeling.ORDER),
        EnumHyperparameter(
            name="order_method",
            enum=FixedOrderMethod,
            default=FixedOrderMethod.ORIGINAL_ORDER,
            depends_on=("modeling", DpModeling.ORDER),
        ),
    ]
    nodes_convention: list
    transitions: dict
    modeling: DpModeling

    def init_model(self, **kwargs: Any) -> None:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        if kwargs["modeling"] == DpModeling.ORDER:
            self.init_in_order(**kwargs)
        if kwargs["modeling"] == DpModeling.ANY_ORDER:
            self.init_any_order(**kwargs)
        self.modeling = kwargs["modeling"]

    def init_in_order(self, **kwargs: Any):
        model = dp.Model(maximize=True)
        indexes = range(self.problem.number_nodes)
        if kwargs["order_method"] == FixedOrderMethod.DEGREE:
            degrees = nx.degree_centrality(self.problem.graph_nx)
            sorted_nodes = sorted(degrees, key=lambda x: degrees[x], reverse=True)
            sorted_nodes = [self.problem.nodes_to_index[n] for n in sorted_nodes]
        else:
            sorted_nodes = [
                self.problem.nodes_to_index[p] for p in sorted(self.problem.nodes)
            ]
        self.nodes_convention = sorted_nodes
        original_node_to_node_convention = {
            sorted_nodes[i]: i for i in range(len(sorted_nodes))
        }
        nodes = model.add_object_type(number=self.problem.number_nodes)
        open_set = model.add_set_var(object_type=nodes, target=indexes)
        cur_node = model.add_element_var(object_type=nodes, target=0)
        model.add_base_case([open_set.is_empty()])
        to_remove = []
        self.transitions = {}
        for i in range(self.problem.number_nodes):
            n_ = self.nodes_convention[i]
            to_remove.append(
                set(
                    [
                        original_node_to_node_convention[self.problem.nodes_to_index[n]]
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
                (open_set, open_set.difference(nodes_to_remove[cur_node])),
                (cur_node, cur_node + 1),
            ],
            preconditions=[open_set.contains(cur_node)],
        )
        model.add_transition(take)
        self.transitions["pick"] = take
        no_take = dp.Transition(
            name=f"no_pick",
            cost=dp.IntExpr.state_cost(),
            effects=[(open_set, open_set.remove(cur_node)), (cur_node, cur_node + 1)],
            preconditions=[
                # open.contains(cur_node)
            ],
        )
        model.add_transition(no_take)
        self.transitions["no_pick"] = no_take
        self.model = model

    def init_any_order(self, **kwargs: Any) -> None:
        model = dp.Model(maximize=True)
        indexes = range(self.problem.number_nodes)

        nodes = model.add_object_type(number=self.problem.number_nodes)
        open_set = model.add_set_var(object_type=nodes, target=indexes)
        model.add_base_case([open_set.is_empty()])
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
        self.transitions = {}
        for i in range(self.problem.number_nodes):
            take = dp.Transition(
                name=f"pick_{i}",
                cost=1 + dp.IntExpr.state_cost(),
                effects=[(open_set, open_set.difference(nodes_to_remove[i]))],
                preconditions=[open_set.contains(i)],
            )
            model.add_transition(take)
            self.transitions[("pick", i)] = take
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

    def set_warm_start(self, solution: MisSolution) -> None:
        transitions = []
        if self.modeling == DpModeling.ORDER:
            for i_node in self.nodes_convention:
                if solution.chosen[i_node]:
                    transitions.append(self.transitions["pick"])
                else:
                    transitions.append(self.transitions["no_pick"])
            self.initial_solution = transitions
        if self.modeling == DpModeling.ANY_ORDER:
            self.initial_solution = [
                self.transitions[("pick", i)]
                for i in range(self.problem.number_nodes)
                if solution.chosen[i] == 1
            ]
