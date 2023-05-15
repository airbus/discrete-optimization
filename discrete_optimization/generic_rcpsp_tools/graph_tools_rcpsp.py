#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Union

import networkx as nx
from networkx import NetworkXNoCycle

from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel
from discrete_optimization.rcpsp.rcpsp_model_preemptive import RCPSPModelPreemptive
from discrete_optimization.rcpsp.special_constraints import (
    SpecialConstraintsDescription,
)
from discrete_optimization.rcpsp.specialized_rcpsp.rcpsp_specialized_constraints import (
    RCPSPModelSpecialConstraintsPreemptive,
)
from discrete_optimization.rcpsp_multiskill.rcpsp_multiskill import (
    MS_RCPSPModel,
    MS_RCPSPModel_Variant,
)


class GraphRCPSP:
    def __init__(
        self,
        problem: Union[
            RCPSPModel,
            RCPSPModelPreemptive,
            MS_RCPSPModel,
            MS_RCPSPModel_Variant,
        ],
    ):
        self.problem = problem
        self.graph = self.problem.compute_graph()
        self.graph_nx = self.graph.graph_nx
        self.ancestors_map = {}
        self.descendants_map = {}
        descendants = {
            n: nx.algorithms.descendants(self.graph_nx, n)
            for n in self.graph_nx.nodes()
        }
        self.source = self.problem.source_task
        self.sink = self.problem.sink_task
        self.all_activities = set(self.problem.tasks_list)
        for k in descendants:
            self.descendants_map[k] = {
                "succs": descendants[k],
                "nb": len(descendants[k]),
            }
        ancestors = {
            n: nx.algorithms.ancestors(self.graph_nx, n) for n in self.graph_nx.nodes()
        }
        for k in ancestors:
            self.ancestors_map[k] = {"succs": ancestors[k], "nb": len(ancestors[k])}
        self.graph_without_source_sink = nx.subgraph(
            self.graph_nx,
            [
                n
                for n in self.graph_nx
                if n not in {self.problem.sink_task, self.problem.source_task}
            ],
        )

    def get_next_activities(self, task):
        return self.graph.get_neighbors(task)

    def get_pred_activities(self, task):
        return self.graph.get_predecessors(task)

    def get_descendants_activities(self, task):
        return self.descendants_map.get(task, {"succs": set()})["succs"]

    def get_ancestors_activities(self, task):
        return self.ancestors_map.get(task, {"succs": set()})["succs"]

    def check_loop(self):
        try:
            cycles = nx.find_cycle(self.graph_nx, orientation="original")
        except NetworkXNoCycle:
            cycles = None
        return cycles

    def compute_component_in_non_dummy_graph(self):
        return [
            c
            for c in sorted(
                nx.weakly_connected_components(self.graph_without_source_sink),
                key=len,
                reverse=True,
            )
        ]


class GraphRCPSPSpecialConstraints(GraphRCPSP):
    def __init__(
        self,
        problem: Union[RCPSPModel, RCPSPModelSpecialConstraintsPreemptive],
    ):
        if isinstance(problem, RCPSPModel) and not problem.do_special_constraints:
            raise ValueError("this graph is meant for models with special constraints")
        super().__init__(problem)
        self.special_constraints: SpecialConstraintsDescription = (
            problem.special_constraints
        )
        self.graph_constraints = nx.DiGraph()
        for K in (
            self.special_constraints.start_together
            + self.special_constraints.start_at_end
            + self.special_constraints.start_at_end_plus_offset
            + self.special_constraints.start_after_nunit
            + self.special_constraints.disjunctive_tasks
        ):
            t1 = K[0]
            t2 = K[1]
            if t1 not in self.graph_constraints:
                self.graph_constraints.add_node(t1)
            if t2 not in self.graph_constraints:
                self.graph_constraints.add_node(t2)
            self.graph_constraints.add_edge(t1, t2)
            self.graph_constraints.add_edge(t2, t1)
        self.components_graph_constraints = [
            c for c in nx.strongly_connected_components(self.graph_constraints)
        ]
        self.index_components = {}
        for i in range(len(self.components_graph_constraints)):
            for ci in self.components_graph_constraints[i]:
                self.index_components[ci] = i

    def get_neighbors_constraints(self, task):
        if task in self.graph_constraints:
            return list(self.graph_constraints.neighbors(task))
        return []

    def get_pred_constraints(self, task):
        if task in self.graph_constraints:
            return list(self.graph_constraints.predecessors(task))
        return []


def build_unrelated_task(graph: GraphRCPSP):
    ancestors = graph.ancestors_map
    descendants = graph.descendants_map
    all_tasks = set(descendants)
    unrel = {
        n: all_tasks.difference(
            set(ancestors[n]["succs"]).union(set(descendants[n]["succs"])).union({n})
        )
        for n in all_tasks
    }
    set_pairs = set()
    for task in unrel:
        for other_task in unrel[task]:
            if (task, other_task) not in set_pairs and (
                other_task,
                task,
            ) not in set_pairs:
                set_pairs.add((task, other_task))
    return unrel, set_pairs


def build_graph_rcpsp_object(rcpsp_problem: Union[RCPSPModel, RCPSPModelPreemptive]):
    if (
        hasattr(rcpsp_problem, "do_special_constraints")
        and rcpsp_problem.do_special_constraints
    ):
        return GraphRCPSPSpecialConstraints(problem=rcpsp_problem)
    else:
        return GraphRCPSP(problem=rcpsp_problem)
