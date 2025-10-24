#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from collections import defaultdict
from typing import Any, Optional

import networkx as nx

from discrete_optimization.binpack.problem import BinPackProblem, BinPackSolution
from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_solver import SolverDO
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
        self.graph = graph
        self.neighbors = {n: set(self.graph.neighbors(n)) for n in self.graph.nodes}

    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        callback = CallbackList(callbacks=callbacks)
        callback.on_solve_start(self)
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
                for bin_ in used_bins:
                    if weight + capa_per_bins[bin_] > self.problem.capacity_bin:
                        continue
                    if any(n in item_per_bins[bin_] for n in neighbors):
                        continue
                    capa_per_bins[bin_] += weight
                    allocation[j] = bin_
                    item_per_bins[bin_].add(j)
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
        callback.on_solve_end(res, self)
        return res


class GreedyBinPackOpenEvolve(SolverDO):
    problem: BinPackProblem

    def __init__(self, problem: BinPackProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)
        graph = nx.Graph()
        graph.add_nodes_from(range(self.problem.nb_items))
        if self.problem.has_constraint:
            graph.add_edges_from(list(self.problem.incompatible_items))
        self.graph = graph
        self.neighbors = {n: set(self.graph.neighbors(n)) for n in self.graph.nodes}

    def solve(self, callbacks: Optional[list[Callback]] = None, **kwargs):
        """
        - list_weights: give for each item its size.
        - set_conflicts: {(index1, index2), (index0, index4)...} : means that index1 and index2 are not put in the same bin etc
        - capacity_bin : is the size of each bin
        Returns a list of int (corresponding to bin index for each item index), the len of the list should be len(list_weights).
        """
        callback = CallbackList(callbacks=callbacks)
        callback.on_solve_start(self)

        list_weights = [item.weight for item in self.problem.list_items]
        n = self.problem.nb_items
        bin_assignment = [0] * n
        bin_weights = [0]  # Keeps track of weights in each bin.
        bin_conflicts = [
            set()
        ]  # Keeps track of items in each bin for conflict checking.

        # Heuristic: Sort by decreasing weight, then decreasing number of conflicts
        sorted_items = sorted(
            range(n),
            key=lambda i: (list_weights[i], -len(self.neighbors[i])),
            reverse=True,
        )

        for i in sorted_items:
            best_bin = -1
            min_weight_increase = float("inf")  # Minimizes the increase in bin weight
        bin_conflicts = [
            set()
        ]  # Keeps track of items in each bin for conflict checking.

        # Heuristic: Sort by decreasing weight, then decreasing number of conflicts
        sorted_items = sorted(
            range(n),
            key=lambda i: (list_weights[i], -len(self.neighbors[i])),
            reverse=True,
        )

        for i in sorted_items:
            best_bin = -1
            min_weight_increase = float("inf")
            for j in range(len(bin_weights)):
                valid_placement = True
                for k in bin_conflicts[j]:
                    if k in self.neighbors[i]:
                        valid_placement = False
                        break
                if (
                    valid_placement
                    and bin_weights[j] + list_weights[i] <= self.problem.capacity_bin
                ):
                    if bin_weights[j] + list_weights[i] < min_weight_increase:
                        min_weight_increase = bin_weights[j] + list_weights[i]
                        best_bin = j

            if best_bin != -1:
                bin_assignment[i] = best_bin
                bin_weights[best_bin] += list_weights[i]
                bin_conflicts[best_bin].add(i)
            else:
                bin_assignment[i] = len(bin_weights)
                bin_weights.append(list_weights[i])
                bin_conflicts.append({i})
        sol = BinPackSolution(problem=self.problem, allocation=bin_assignment)
        fit = self.aggreg_from_sol(sol)
        res = self.create_result_storage([(sol, fit)])
        callback.on_solve_end(res, self)
        return res
