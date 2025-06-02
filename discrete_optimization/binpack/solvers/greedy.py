#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from collections import defaultdict
from enum import Enum
from typing import Any, Optional, Union

import networkx as nx

from discrete_optimization.binpack.problem import BinPackProblem, BinPackSolution
from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    EnumHyperparameter,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)


class GreedyBinPackSolver(SolverDO):
    problem: BinPackProblem

    def __init__(self, problem: BinPackProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)
        graph = nx.Graph()
        graph.add_nodes_from(range(self.problem.nb_items))
        if self.problem.has_constraint:
            graph.add_edges_from(list(self.problem.incompatible_items))
            neighs = graph.adj
        self.graph = graph
        self.neighbors = {n: set(self.graph.neighbors(n)) for n in self.graph.nodes}

    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        allocation = [None for i in range(self.problem.nb_items)]
        capa_per_bins = defaultdict(lambda: 0)
        used_bins = []
        item_per_bins = defaultdict(lambda: set())
        nb_bins = 0
        for j in range(self.problem.nb_items):
            weight = self.problem.list_items[j].weight
            neighbors = self.neighbors[j]
            if nb_bins == 0:
                nb_bins = 1
                used_bins.append(0)
                capa_per_bins[0] += weight
                allocation[j] = 0
                item_per_bins[0].add(j)
            else:
                success = False
                for bin in used_bins:
                    if weight + capa_per_bins[bin] > self.problem.capacity_bin:
                        continue
                    if any(n in item_per_bins[bin] for n in neighbors):
                        continue
                    capa_per_bins[bin] += weight
                    allocation[j] = bin
                    item_per_bins[bin].add(j)
                    success = True
                    break
                if not success:
                    new_bin = used_bins[-1] + 1
                    used_bins.append(new_bin)
                    item_per_bins[new_bin].add(j)
                    capa_per_bins[new_bin] += weight
                    allocation[j] = new_bin
        res = self.create_result_storage([])
        sol = BinPackSolution(problem=self.problem, allocation=allocation)
        fit = self.aggreg_from_sol(sol)
        res.append((sol, fit))
        return res
