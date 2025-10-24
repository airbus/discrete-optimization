#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from collections import defaultdict
from copy import deepcopy
from dataclasses import field

from discrete_optimization.generic_tools.do_problem import *


class BinPackSolution(Solution):
    def __init__(self, problem: "BinPackProblem", allocation: list[int]):
        self.problem = problem
        self.allocation = allocation

    def copy(self) -> "BinPackSolution":
        return BinPackSolution(
            problem=self.problem, allocation=deepcopy(self.allocation)
        )

    def change_problem(self, new_problem: "Problem") -> None:
        self.problem = new_problem


@dataclass(frozen=True)
class ItemBinPack:
    index: int = field(init=True)
    weight: float = field(init=True)

    def __str__(self) -> str:
        return "ind: " + str(self.index) + " weight: " + str(self.weight)


class BinPackProblem(Problem):
    def __init__(
        self,
        list_items: list[ItemBinPack],
        capacity_bin: int,
        incompatible_items: set[tuple[int, int]] = None,
    ):
        self.list_items = list_items
        self.nb_items = len(self.list_items)
        self.capacity_bin = capacity_bin
        self.incompatible_items = incompatible_items
        self.has_constraint = not (
            incompatible_items is None or len(incompatible_items) == 0
        )

    def evaluate(self, variable: BinPackSolution) -> dict[str, float]:
        nb_bins = len(set(variable.allocation))
        sat = self.satisfy(variable)
        penalty = 10000 if not sat else 0
        return {"nb_bins": nb_bins, "penalty": penalty}

    def compute_weights(self, variable: BinPackSolution) -> dict[int, float]:
        weight_per_bins = defaultdict(lambda: 0)
        for i in range(self.nb_items):
            weight_per_bins[variable.allocation[i]] += self.list_items[i].weight
        return weight_per_bins

    def satisfy(self, variable: BinPackSolution) -> bool:
        weight_per_bins = defaultdict(lambda: 0)
        for i in range(self.nb_items):
            weight_per_bins[variable.allocation[i]] += self.list_items[i].weight
            if weight_per_bins[variable.allocation[i]] > self.capacity_bin:
                print("capa")
                return False
        if self.has_constraint:
            for i, j in self.incompatible_items:
                if variable.allocation[i] == variable.allocation[j]:
                    print("conflict")
                    return False
        return True

    def get_attribute_register(self) -> EncodingRegister:
        return EncodingRegister(
            dict_attribute_to_type={
                "allocation": {
                    "type": TypeAttribute.LIST_INTEGER,
                    "up": self.nb_items,
                    "low": 0,
                }
            }
        )

    def get_solution_type(self) -> type[Solution]:
        return BinPackSolution

    def get_objective_register(self) -> ObjectiveRegister:
        return ObjectiveRegister(
            objective_sense=ModeOptim.MINIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc={
                "nb_bins": ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=1),
                "penalty": ObjectiveDoc(type=TypeObjective.PENALTY, default_weight=1),
            },
        )
