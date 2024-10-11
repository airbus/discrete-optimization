#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional, Union, cast

import numpy as np

from discrete_optimization.generic_tools.do_problem import (
    EncodingRegister,
    MethodAggregating,
    ModeOptim,
    ObjectiveDoc,
    ObjectiveHandling,
    ObjectiveRegister,
    Problem,
    RobustProblem,
    Solution,
    TupleFitness,
    TypeAttribute,
    TypeObjective,
)


@dataclass(frozen=True)
class Item:
    index: int
    value: float
    weight: float

    def __str__(self) -> str:
        return (
            "ind: "
            + str(self.index)
            + " weight: "
            + str(self.weight)
            + " value: "
            + str(self.value)
        )


class KnapsackSolution(Solution):
    def __init__(
        self,
        problem: "KnapsackProblem",
        list_taken: list[int],
        value: Optional[float] = None,
        weight: Optional[float] = None,
    ):
        self.problem = problem
        self.value = value
        self.weight = weight
        self.list_taken = list_taken

    def copy(self) -> "KnapsackSolution":
        return KnapsackSolution(
            problem=self.problem,
            value=self.value,
            weight=self.weight,
            list_taken=list(self.list_taken),
        )

    def lazy_copy(self) -> "KnapsackSolution":
        return KnapsackSolution(
            problem=self.problem,
            value=self.value,
            weight=self.weight,
            list_taken=self.list_taken,
        )

    def change_problem(self, new_problem: Problem) -> None:
        if not isinstance(new_problem, KnapsackProblem):
            raise ValueError("new_problem must a KnapsackModel for a KnapsackSolution.")
        self.problem = new_problem
        self.list_taken = list(self.list_taken)

    def __str__(self) -> str:
        s = "Value=" + str(self.value) + "\n"
        s += "Weight=" + str(self.weight) + "\n"
        s += "Taken : " + str(self.list_taken)
        return s

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, KnapsackSolution) and self.list_taken == other.list_taken
        )


class KnapsackProblem(Problem):
    def __init__(
        self,
        list_items: list[Item],
        max_capacity: float,
        force_recompute_values: bool = False,
    ):
        self.list_items = list_items
        self.nb_items = len(list_items)
        self.max_capacity = max_capacity
        self.index_to_item = {
            list_items[i].index: list_items[i] for i in range(self.nb_items)
        }
        self.index_to_index_list = {
            list_items[i].index: i for i in range(self.nb_items)
        }
        self.force_recompute_values = force_recompute_values

    def get_attribute_register(self) -> EncodingRegister:
        dict_register = {
            "list_taken": {
                "name": "list_taken",
                "type": [TypeAttribute.LIST_BOOLEAN, TypeAttribute.LIST_BOOLEAN_KNAP],
                "n": self.nb_items,
            }
        }
        return EncodingRegister(dict_register)

    def get_objective_register(self) -> ObjectiveRegister:
        dict_objective = {
            "weight_violation": ObjectiveDoc(
                type=TypeObjective.PENALTY, default_weight=-100000.0
            ),
            "value": ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=1.0),
        }
        return ObjectiveRegister(
            objective_sense=ModeOptim.MAXIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc=dict_objective,
        )

    def evaluate_from_encoding(
        self, int_vector: list[int], encoding_name: str
    ) -> dict[str, float]:
        if encoding_name == "list_taken":
            kp_sol = KnapsackSolution(problem=self, list_taken=int_vector)
        else:
            raise NotImplementedError("encoding_name must be 'list_taken'")
        objectives = self.evaluate(kp_sol)
        return objectives

    def evaluate(self, knapsack_solution: KnapsackSolution) -> dict[str, float]:  # type: ignore # avoid isinstance checks for efficiency
        if knapsack_solution.value is None or self.force_recompute_values:
            val = self.evaluate_value(knapsack_solution)
        else:
            val = knapsack_solution.value
        w_violation = self.evaluate_weight_violation(knapsack_solution)
        return {"value": val, "weight_violation": w_violation}

    def evaluate_value(self, knapsack_solution: KnapsackSolution) -> float:
        s = sum(
            [
                knapsack_solution.list_taken[i] * self.list_items[i].value
                for i in range(self.nb_items)
            ]
        )
        w = sum(
            [
                knapsack_solution.list_taken[i] * self.list_items[i].weight
                for i in range(self.nb_items)
            ]
        )
        knapsack_solution.value = s
        knapsack_solution.weight = w
        return sum(
            [
                knapsack_solution.list_taken[i] * self.list_items[i].value
                for i in range(self.nb_items)
            ]
        )

    def evaluate_weight_violation(self, knapsack_solution: KnapsackSolution) -> float:
        return max(0.0, knapsack_solution.weight - self.max_capacity)  # type: ignore  # avoid is None check for efficiency

    def satisfy(self, knapsack_solution: KnapsackSolution) -> bool:  # type: ignore  # avoid isinstance checks for efficiency
        if knapsack_solution.value is None:
            self.evaluate(knapsack_solution)
        return knapsack_solution.weight <= self.max_capacity  # type: ignore  # avoid is None check for efficiency

    def __str__(self) -> str:
        s = (
            "Knapsack model with "
            + str(self.nb_items)
            + " items and capacity "
            + str(self.max_capacity)
            + "\n"
        )
        s += "\n".join([str(item) for item in self.list_items])
        return s

    def get_dummy_solution(self) -> KnapsackSolution:
        kp_sol = KnapsackSolution(problem=self, list_taken=[0] * self.nb_items)
        self.evaluate(kp_sol)
        return kp_sol

    def get_solution_type(self) -> type[Solution]:
        return KnapsackSolution


def create_subknapsack_problem(
    knapsack_problem: KnapsackProblem,
    solution: KnapsackSolution,
    indexes_to_remove: set[int],
    indexes_to_keep: set[int] = None,
):
    if indexes_to_keep is None:
        indexes_to_keep = set(range(knapsack_problem.nb_items)).difference(
            indexes_to_remove
        )
    weight = sum(
        solution.list_taken[ind] * knapsack_problem.list_items[ind].weight
        for ind in indexes_to_remove
    )
    return KnapsackProblem(
        list_items=[
            knapsack_problem.list_items[ind] for ind in sorted(indexes_to_keep)
        ],
        max_capacity=knapsack_problem.max_capacity - weight,
    )


class MobjKnapsackModel(KnapsackProblem):
    @staticmethod
    def from_knapsack(knapsack_problem: KnapsackProblem) -> "MobjKnapsackModel":
        return MobjKnapsackModel(
            list_items=knapsack_problem.list_items,
            max_capacity=knapsack_problem.max_capacity,
            force_recompute_values=knapsack_problem.force_recompute_values,
        )

    def get_objective_register(self) -> ObjectiveRegister:
        dict_objective = {
            "weight_violation": ObjectiveDoc(
                type=TypeObjective.PENALTY, default_weight=-100000.0
            ),
            "heaviest_item": ObjectiveDoc(
                type=TypeObjective.OBJECTIVE, default_weight=1.0
            ),
            "weight": ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=-1.0),
            "value": ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=1.0),
        }
        return ObjectiveRegister(
            objective_sense=ModeOptim.MAXIMIZATION,
            objective_handling=ObjectiveHandling.MULTI_OBJ,
            dict_objective_to_doc=dict_objective,
        )

    def evaluate(self, knapsack_solution: KnapsackSolution) -> dict[str, float]:  # type: ignore  # avoid isinstance checks for efficiency
        res = super().evaluate(knapsack_solution)
        heaviest = 0.0
        weight = 0.0
        for i in range(self.nb_items):
            if knapsack_solution.list_taken[i] == 1:
                heaviest = max(heaviest, self.list_items[i].weight)
                weight += self.list_items[i].weight
        res["heaviest_item"] = heaviest
        res["weight"] = weight
        return res

    def evaluate_mobj_from_dict(self, dict_values: dict[str, float]) -> TupleFitness:
        return TupleFitness(
            np.array([dict_values["value"], -dict_values["heaviest_item"]]), 2
        )

    def evaluate_mobj(self, solution: KnapsackSolution) -> TupleFitness:  # type: ignore  # avoid isinstance checks for efficiency
        return self.evaluate_mobj_from_dict(self.evaluate(solution))


class MultidimensionalKnapsackSolution(Solution):
    def __init__(
        self,
        problem: Union[
            "MultidimensionalKnapsackProblem",
            "MultiScenarioMultidimensionalKnapsackProblem",
        ],
        list_taken: list[int],
        value: Optional[float] = None,
        weights: Optional[list[float]] = None,
    ):
        self.problem = problem
        self.value = value
        self.weights = weights
        self.list_taken = list_taken

    def copy(self) -> "MultidimensionalKnapsackSolution":
        return MultidimensionalKnapsackSolution(
            problem=self.problem,
            value=self.value,
            weights=self.weights,
            list_taken=list(self.list_taken),
        )

    def lazy_copy(self) -> "MultidimensionalKnapsackSolution":
        return MultidimensionalKnapsackSolution(
            problem=self.problem,
            value=self.value,
            weights=self.weights,
            list_taken=self.list_taken,
        )

    def change_problem(self, new_problem: Problem) -> None:
        if not isinstance(new_problem, MultidimensionalKnapsackProblem):
            raise ValueError(
                "new_problem must a MultidimensionalKnapsack for a KnapsackSolutionMultidimensional."
            )
        self.problem = new_problem
        self.list_taken = list(self.list_taken)

    def __str__(self) -> str:
        s = "Value=" + str(self.value) + "\n"
        s += "Weights=" + str(self.weights) + "\n"
        s += "Taken : " + str(self.list_taken)
        return s

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, MultidimensionalKnapsackSolution)
            and self.list_taken == other.list_taken
        )


@dataclass(frozen=True)
class ItemMultidimensional:
    index: int
    value: float
    weights: list[float]

    def __str__(self) -> str:
        return (
            "ind: "
            + str(self.index)
            + " weights: "
            + str(self.weights)
            + " value: "
            + str(self.value)
        )


class MultidimensionalKnapsackProblem(Problem):
    def __init__(
        self,
        list_items: list[ItemMultidimensional],
        max_capacities: list[float],
        force_recompute_values: bool = False,
    ):
        self.list_items = list_items
        self.nb_items = len(list_items)
        self.max_capacities = max_capacities
        self.index_to_item = {
            list_items[i].index: list_items[i] for i in range(self.nb_items)
        }
        self.index_to_index_list = {
            list_items[i].index: i for i in range(self.nb_items)
        }
        self.force_recompute_values = force_recompute_values

    def get_attribute_register(self) -> EncodingRegister:
        dict_register = {
            "list_taken": {
                "name": "list_taken",
                "type": [TypeAttribute.LIST_BOOLEAN],
                "n": self.nb_items,
            }
        }
        return EncodingRegister(dict_register)

    def get_objective_register(self) -> ObjectiveRegister:
        dict_objective = {
            "weight_violation": ObjectiveDoc(
                type=TypeObjective.PENALTY, default_weight=-100000.0
            ),
            "value": ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=1.0),
        }
        return ObjectiveRegister(
            objective_sense=ModeOptim.MAXIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc=dict_objective,
        )

    def evaluate_from_encoding(
        self, int_vector: list[int], encoding_name: str
    ) -> dict[str, float]:
        if encoding_name == "list_taken":
            kp_sol = MultidimensionalKnapsackSolution(
                problem=self, list_taken=int_vector
            )
        elif encoding_name == "custom":
            kwargs: dict[str, Any] = {encoding_name: int_vector}
            kp_sol = MultidimensionalKnapsackSolution(problem=self, **kwargs)
        else:
            raise NotImplementedError("encoding_name must be 'list_taken' or 'custom'")
        objectives = self.evaluate(kp_sol)
        return objectives

    def evaluate(self, knapsack_solution: MultidimensionalKnapsackSolution) -> dict[str, float]:  # type: ignore  # avoid isinstance checks for efficiency
        if knapsack_solution.value is None or self.force_recompute_values:
            val = self.evaluate_value(knapsack_solution)
        else:
            val = knapsack_solution.value
        w_violation = self.evaluate_weight_violation(knapsack_solution)
        return {"value": val, "weight_violation": w_violation}

    def evaluate_value(
        self, knapsack_solution: MultidimensionalKnapsackSolution
    ) -> float:
        s = sum(
            [
                knapsack_solution.list_taken[i] * self.list_items[i].value
                for i in range(self.nb_items)
            ]
        )
        w = [
            sum(
                [
                    knapsack_solution.list_taken[i] * self.list_items[i].weights[j]
                    for i in range(self.nb_items)
                ]
            )
            for j in range(len(self.max_capacities))
        ]
        knapsack_solution.value = s
        knapsack_solution.weights = w
        return s

    def evaluate_weight_violation(
        self, knapsack_solution: MultidimensionalKnapsackSolution
    ) -> float:
        if knapsack_solution.weights is None:
            raise RuntimeError(
                "knapsack_solution.weights should not be None when calling evaluate_weight_violation."
            )
        return sum(
            [
                max(0.0, knapsack_solution.weights[j] - self.max_capacities[j])
                for j in range(len(self.max_capacities))
            ]
        )

    def satisfy(self, knapsack_solution: MultidimensionalKnapsackSolution) -> bool:  # type: ignore  # avoid isinstance checks for efficiency
        if knapsack_solution.value is None or knapsack_solution.weights is None:
            self.evaluate(knapsack_solution)
            if knapsack_solution.value is None or knapsack_solution.weights is None:
                raise RuntimeError(
                    "knapsack_solution.value and knapsack_solution.weights should not be None now."
                )
        return all(
            knapsack_solution.weights[j] <= self.max_capacities[j]
            for j in range(len(self.max_capacities))
        )

    def __str__(self) -> str:
        s = (
            "Knapsack model with "
            + str(self.nb_items)
            + " items and capacity "
            + str(self.max_capacities)
            + "\n"
        )
        s += "\n".join([str(item) for item in self.list_items])
        return s

    def get_dummy_solution(self) -> MultidimensionalKnapsackSolution:
        kp_sol = MultidimensionalKnapsackSolution(
            problem=self, list_taken=[0] * self.nb_items
        )
        self.evaluate(kp_sol)
        return kp_sol

    def get_solution_type(self) -> type[Solution]:
        return MultidimensionalKnapsackSolution

    def copy(self) -> "MultidimensionalKnapsackProblem":
        return MultidimensionalKnapsackProblem(
            list_items=[deepcopy(x) for x in self.list_items],
            max_capacities=list(self.max_capacities),
            force_recompute_values=self.force_recompute_values,
        )


class MultiScenarioMultidimensionalKnapsackProblem(RobustProblem):
    list_problem: Sequence[MultidimensionalKnapsackProblem]

    def __init__(
        self,
        list_problem: Sequence[MultidimensionalKnapsackProblem],
        method_aggregating: MethodAggregating,
    ):
        super().__init__(list_problem, method_aggregating)

    def get_dummy_solution(self) -> MultidimensionalKnapsackSolution:
        return cast(
            MultidimensionalKnapsackProblem, self.list_problem[0]
        ).get_dummy_solution()


def from_kp_to_multi(
    knapsack_problem: KnapsackProblem,
) -> MultidimensionalKnapsackProblem:
    return MultidimensionalKnapsackProblem(
        list_items=[
            ItemMultidimensional(index=x.index, value=x.value, weights=[x.weight])
            for x in knapsack_problem.list_items
        ],
        max_capacities=[knapsack_problem.max_capacity],
    )


def create_noised_scenario(
    problem: MultidimensionalKnapsackProblem, nb_scenarios: int = 20
) -> list[MultidimensionalKnapsackProblem]:
    scenarios = [problem.copy() for i in range(nb_scenarios)]
    for p in scenarios:
        litem = []
        for item in p.list_items:
            litem += [
                ItemMultidimensional(
                    index=item.index,
                    value=np.random.randint(
                        max(0, int(item.value * 0.9)), int(item.value * 1.1)
                    ),
                    weights=item.weights,
                )
            ]
        p.list_items = litem
    return scenarios
