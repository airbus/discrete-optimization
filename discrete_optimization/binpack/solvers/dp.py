#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import math
import re
from enum import Enum
from functools import reduce
from typing import Any

import didppy as dp

from discrete_optimization.binpack.problem import BinPackProblem, BinPackSolution
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.dyn_prog_tools import DpSolver
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
)


class ModelingDpBinPack(Enum):
    ASSIGN_ITEM_BINS = 0
    PACK_ITEMS = 1


class DpBinPackSolver(DpSolver):
    problem: BinPackProblem
    transitions: dict
    modeling: ModelingDpBinPack
    hyperparameters = DpSolver.hyperparameters + [
        EnumHyperparameter(
            name="modeling",
            enum=ModelingDpBinPack,
            default=ModelingDpBinPack.ASSIGN_ITEM_BINS,
        )
    ]

    def init_model(self, **kwargs):
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        if kwargs["modeling"] == ModelingDpBinPack.ASSIGN_ITEM_BINS:
            self.init_model_assign_item_and_bins(**kwargs)
        if kwargs["modeling"] == ModelingDpBinPack.PACK_ITEMS:
            if self.problem.has_constraint:
                self.init_model_fill_bins_with_constraints(**kwargs)
            else:
                self.init_model_fill_bins(**kwargs)
        self.modeling = kwargs["modeling"]

    def init_model_assign_item_and_bins(self, **kwargs: Any) -> None:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        upper_bound = kwargs.get("upper_bound", self.problem.nb_items)
        nodes_order = [i for i in range(self.problem.nb_items)]
        model = dp.Model()
        bins = [model.add_int_var(target=0)] + [
            model.add_int_var(target=i) for i in range(1, self.problem.nb_items)
        ]
        capacities = [model.add_int_var(target=self.problem.list_items[0].weight)] + [
            model.add_int_var(target=0) for i in range(1, upper_bound)
        ]
        item = model.add_object_type(number=self.problem.nb_items)
        cur_bin = model.add_int_var(target=0)
        unassigned = model.add_set_var(
            object_type=item, target=range(1, self.problem.nb_items)
        )
        model.add_base_case([unassigned.is_empty()])
        self.transitions = {}
        for i in range(1, self.problem.nb_items):
            if self.problem.has_constraint:
                edges = [
                    x for x in self.problem.incompatible_items if x[0] == i or x[1] == i
                ]
                neighs = [x[0] if x[0] != i else x[1] for x in edges]
            else:
                neighs = []
            for bin in range(upper_bound):
                assign = dp.Transition(
                    name=f"{i, bin}",
                    cost=dp.max(bin, cur_bin) - cur_bin + dp.IntExpr.state_cost(),
                    effects=[
                        (unassigned, unassigned.remove(i)),
                        (cur_bin, dp.max(cur_bin, bin)),
                        (bins[i], bin),
                        (
                            capacities[bin],
                            capacities[bin] + self.problem.list_items[i].weight,
                        ),
                    ],
                    preconditions=[
                        bin <= cur_bin + 1,
                        unassigned.contains(i),
                        ~unassigned.contains(i - 1),
                        capacities[bin] + self.problem.list_items[i].weight
                        <= self.problem.capacity_bin,
                    ]
                    + [bins[n] != bin for n in neighs],
                )
                model.add_transition(assign)
                self.transitions[(i, bin)] = assign
        self.model = model

    def init_model_fill_bins(self, **kwargs):
        # Thanks :
        # https://colab.research.google.com/github/domain-independent-dp/didp-rs/blob/main/didppy/examples/bin-packing.ipynb
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        upper_bound = kwargs.get("upper_bound", self.problem.nb_items)
        w = [self.problem.list_items[i].weight for i in range(self.problem.nb_items)]
        c = self.problem.capacity_bin
        n = self.problem.nb_items
        # The weight in the first term of the second bound.
        weight_2_1 = [1 if w[i] > c / 2 else 0 for i in range(n)]
        # The weight in the second term of the second bound.
        weight_2_2 = [1 / 2 if w[i] == c / 2 else 0 for i in range(n)]
        # The weight in the third bound (truncated to three decimal points).
        weight_3 = [
            1.0
            if w[i] > c * 2 / 3
            else 2 / 3 // 0.001 * 1000
            if w[i] == c * 2 / 3
            else 0.5
            if w[i] > c / 3
            else 1 / 3 // 0.001 * 1000
            if w[i] == c / 3
            else 0.0
            for i in range(n)
        ]
        model = dp.Model()
        item = model.add_object_type(number=n)

        # U
        unpacked = model.add_set_var(object_type=item, target=list(range(n)))
        # r
        remaining = model.add_int_resource_var(target=0, less_is_better=False)
        # k (we want to compare the number of bins with the index of an item)
        number_of_bins = model.add_element_resource_var(
            object_type=item,
            target=0,
            less_is_better=True,
        )

        weight = model.add_int_table(w)

        for i in range(n):
            pack = dp.Transition(
                name="continue pack {}".format(i),
                cost=dp.IntExpr.state_cost(),
                effects=[
                    (unpacked, unpacked.remove(i)),
                    (remaining, remaining - weight[i]),
                ],
                preconditions=[
                    unpacked.contains(i),
                    weight[i] <= remaining,
                    i + 1 >= number_of_bins,
                ],
            )
            model.add_transition(pack)

        for i in range(n):
            open_new = dp.Transition(
                name="open a new bin and pack {}".format(i),
                cost=1 + dp.IntExpr.state_cost(),
                effects=[
                    (unpacked, unpacked.remove(i)),
                    (remaining, c - weight[i]),
                    (number_of_bins, number_of_bins + 1),
                ],
                preconditions=[
                    unpacked.contains(i),
                    i >= number_of_bins,
                    weight[i] > remaining,
                ]
                + [
                    ~unpacked.contains(j) | (weight[j] > remaining)
                    for j in range(n)
                    if i != j
                ],
            )
            model.add_transition(open_new, forced=True)

        model.add_base_case([unpacked.is_empty()])

        model.add_dual_bound(math.ceil((weight[unpacked] - remaining) / c))

        weight_2_1_table = model.add_int_table(weight_2_1)
        weight_2_2_table = model.add_float_table(weight_2_2)
        model.add_dual_bound(
            weight_2_1_table[unpacked]
            + math.ceil(weight_2_2_table[unpacked])
            - (remaining >= c / 2).if_then_else(1, 0)
        )

        weight_3_table = model.add_float_table(weight_3)
        model.add_dual_bound(
            math.ceil(weight_3_table[unpacked])
            - (remaining >= c / 3).if_then_else(1, 0)
        )
        self.model = model

    def init_model_fill_bins_with_constraints(self, **kwargs):
        # Thanks :
        # https://colab.research.google.com/github/domain-independent-dp/didp-rs/blob/main/didppy/examples/bin-packing.ipynb
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        upper_bound = kwargs.get("upper_bound", self.problem.nb_items)
        w = [self.problem.list_items[i].weight for i in range(self.problem.nb_items)]
        c = self.problem.capacity_bin
        n = self.problem.nb_items
        # The weight in the first term of the second bound.
        weight_2_1 = [1 if w[i] > c / 2 else 0 for i in range(n)]
        # The weight in the second term of the second bound.
        weight_2_2 = [1 / 2 if w[i] == c / 2 else 0 for i in range(n)]
        # The weight in the third bound (truncated to three decimal points).
        weight_3 = [
            1.0
            if w[i] > c * 2 / 3
            else 2 / 3 // 0.001 * 1000
            if w[i] == c * 2 / 3
            else 0.5
            if w[i] > c / 3
            else 1 / 3 // 0.001 * 1000
            if w[i] == c / 3
            else 0.0
            for i in range(n)
        ]
        model = dp.Model()
        item = model.add_object_type(number=n)

        # U
        unpacked = model.add_set_var(object_type=item, target=list(range(n)))
        # r
        remaining = model.add_int_resource_var(target=0, less_is_better=False)
        # k (we want to compare the number of bins with the index of an item)
        number_of_bins = model.add_element_resource_var(
            object_type=item,
            target=0,
            less_is_better=True,
        )
        n_bin = model.add_int_var(target=0)

        weight = model.add_int_table(w)
        allocation = [model.add_int_var(target=2 * i) for i in range(n)]
        neighs_dict = {}
        for i in range(n):
            edges = [
                x for x in self.problem.incompatible_items if x[0] == i or x[1] == i
            ]
            neighs = [x[0] if x[0] != i else x[1] for x in edges]
            neighs_dict[i] = neighs

        for i in range(n):
            pack = dp.Transition(
                name="continue pack {}".format(i),
                cost=dp.IntExpr.state_cost(),
                effects=[
                    (unpacked, unpacked.remove(i)),
                    (remaining, remaining - weight[i]),
                    (allocation[i], n_bin),
                ],
                preconditions=[
                    unpacked.contains(i),
                    weight[i] <= remaining,
                    i + 1 >= number_of_bins,
                ]
                + [allocation[neigh] != n_bin for neigh in neighs_dict[i]],
            )
            model.add_transition(pack)

        for i in range(n):
            open_new = dp.Transition(
                name="open a new bin and pack {}".format(i),
                cost=1 + dp.IntExpr.state_cost(),
                effects=[
                    (unpacked, unpacked.remove(i)),
                    (remaining, c - weight[i]),
                    (allocation[i], n_bin + 1),
                    (n_bin, n_bin + 1),
                    (number_of_bins, number_of_bins + 1),
                ],
                preconditions=[
                    unpacked.contains(i),
                    i >= number_of_bins,
                    weight[i] > remaining,
                ]
                + [
                    ~unpacked.contains(j)
                    | (weight[j] > remaining)  # | reduce(lambda x,y:
                    #         x | (allocation[y] == n_bin),
                    #         neighs_dict[j],
                    #         n_bin>=0)
                    for j in range(n)
                    if i != j
                ],
            )
            model.add_transition(open_new, forced=False)

        model.add_base_case([unpacked.is_empty()])

        model.add_dual_bound(math.ceil((weight[unpacked] - remaining) / c))

        weight_2_1_table = model.add_int_table(weight_2_1)
        weight_2_2_table = model.add_float_table(weight_2_2)
        model.add_dual_bound(
            weight_2_1_table[unpacked]
            + math.ceil(weight_2_2_table[unpacked])
            - (remaining >= c / 2).if_then_else(1, 0)
        )

        weight_3_table = model.add_float_table(weight_3)
        model.add_dual_bound(
            math.ceil(weight_3_table[unpacked])
            - (remaining >= c / 3).if_then_else(1, 0)
        )
        self.model = model

    def retrieve_solution_pack_items(self, sol: dp.Solution) -> Solution:
        def extract_numbers(s):
            return tuple(int(num) for num in re.findall(r"\d+", s))

        allocation = [0 for _ in range(self.problem.nb_items)]
        current_bin_packing = 0
        current_item = 0
        for i, t in enumerate(sol.transitions):
            if "open a new bin" in t.name:
                current_bin_packing += 1
                current_item = extract_numbers(t.name)[0]
                allocation[current_item] = current_bin_packing
            if "continue pack" in t.name:
                current_item = extract_numbers(t.name)[0]
                allocation[current_item] = current_bin_packing
        return BinPackSolution(problem=self.problem, allocation=allocation)

    def retrieve_solution_color_transition(self, sol: dp.Solution) -> Solution:
        def extract_numbers(s):
            return tuple(int(num) for num in re.findall(r"\d+", s))

        allocation = [0 for _ in range(self.problem.nb_items)]
        ind = 0
        for t in sol.transitions:
            item, bin = extract_numbers(t.name)
            allocation[item] = bin
        return BinPackSolution(problem=self.problem, allocation=allocation)

    def retrieve_solution(self, sol: dp.Solution) -> Solution:
        if self.modeling == ModelingDpBinPack.PACK_ITEMS:
            return self.retrieve_solution_pack_items(sol)
        if self.modeling == ModelingDpBinPack.ASSIGN_ITEM_BINS:
            return self.retrieve_solution_color_transition(sol)
