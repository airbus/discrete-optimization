from copy import deepcopy
from typing import Dict, List, NamedTuple

import numpy as np

from discrete_optimization.generic_tools.do_problem import (
    EncodingRegister,
    MethodAggregating,
    ModeOptim,
    ObjectiveHandling,
    ObjectiveRegister,
    Problem,
    RobustProblem,
    Solution,
    TupleFitness,
    TypeAttribute,
    TypeObjective,
)


class Item(NamedTuple):
    index: int
    value: float
    weight: float

    def __str__(self):
        return (
            "ind: "
            + str(self.index)
            + " weight: "
            + str(self.weight)
            + " value: "
            + str(self.value)
        )


class KnapsackSolution(Solution):
    value: float
    weight: float
    list_taken: List[bool]

    def __init__(self, problem, list_taken, value=None, weight=None):
        self.problem = problem
        self.value = value
        self.weight = weight
        self.list_taken = list_taken

    def copy(self):
        return KnapsackSolution(
            problem=self.problem,
            value=self.value,
            weight=self.weight,
            list_taken=list(self.list_taken),
        )

    def lazy_copy(self):
        return KnapsackSolution(
            problem=self.problem,
            value=self.value,
            weight=self.weight,
            list_taken=self.list_taken,
        )

    def change_problem(self, new_problem):
        self.__init__(
            problem=new_problem,
            value=self.value,
            weight=self.weight,
            list_taken=list(self.list_taken),
        )

    def __str__(self):
        s = "Value=" + str(self.value) + "\n"
        s += "Weight=" + str(self.weight) + "\n"
        s += "Taken : " + str(self.list_taken)
        return s

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return self.list_taken == other.list_taken


class KnapsackModel(Problem):
    def __init__(
        self,
        list_items: List[Item],
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
            "weight_violation": {
                "type": TypeObjective.PENALTY,
                "default_weight": -100000,
            },
            "value": {"type": TypeObjective.OBJECTIVE, "default_weight": 1},
        }
        return ObjectiveRegister(
            objective_sense=ModeOptim.MAXIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc=dict_objective,
        )

    def evaluate_from_encoding(self, int_vector, encoding_name):
        if encoding_name == "list_taken":
            kp_sol = KnapsackSolution(problem=self, list_taken=int_vector)
        elif encoding_name == "custom":
            kwargs = {encoding_name: int_vector, "problem": self}
            kp_sol = KnapsackSolution(**kwargs)
        objectives = self.evaluate(kp_sol)
        return objectives

    def evaluate(self, knapsack_solution: KnapsackSolution):
        if knapsack_solution.value is None or self.force_recompute_values:
            val = self.evaluate_value(knapsack_solution)
        else:
            val = knapsack_solution.value
        w_violation = self.evaluate_weight_violation(knapsack_solution)
        return {"value": val, "weight_violation": w_violation}

    def evaluate_value(self, knapsack_solution: KnapsackSolution):
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

    def evaluate_weight_violation(self, knapsack_solution: KnapsackSolution):
        return max(0, knapsack_solution.weight - self.max_capacity)

    def satisfy(self, knapsack_solution: KnapsackSolution):
        if knapsack_solution.value is None:
            self.evaluate(knapsack_solution)
        return knapsack_solution.weight <= self.max_capacity

    def __str__(self):
        s = (
            "Knapsack model with "
            + str(self.nb_items)
            + " items and capacity "
            + str(self.max_capacity)
            + "\n"
        )
        s += "\n".join([str(item) for item in self.list_items])
        return s

    def get_dummy_solution(self):
        kp_sol = KnapsackSolution(problem=self, list_taken=[0] * self.nb_items)
        self.evaluate(kp_sol)
        return kp_sol

    def get_solution_type(self):
        return KnapsackSolution


class KnapsackModel_Mobj(KnapsackModel):
    @staticmethod
    def from_knapsack(knapsack_model: KnapsackModel):
        return KnapsackModel_Mobj(
            list_items=knapsack_model.list_items,
            max_capacity=knapsack_model.max_capacity,
            force_recompute_values=knapsack_model.force_recompute_values,
        )

    def get_objective_register(self) -> ObjectiveRegister:
        dict_objective = {
            "weight_violation": {
                "type": TypeObjective.PENALTY,
                "default_weight": -100000,
            },
            "heaviest_item": {"type": TypeObjective.OBJECTIVE, "default_weight": 1},
            "weight": {"type": TypeObjective.OBJECTIVE, "default_weight": -1},
            "value": {"type": TypeObjective.OBJECTIVE, "default_weight": 1},
        }
        return ObjectiveRegister(
            objective_sense=ModeOptim.MAXIMIZATION,
            objective_handling=ObjectiveHandling.MULTI_OBJ,
            dict_objective_to_doc=dict_objective,
        )

    def evaluate(self, knapsack_solution: KnapsackSolution):
        res = super().evaluate(knapsack_solution)
        heaviest = 0
        weight = 0
        for i in range(self.nb_items):
            if knapsack_solution.list_taken[i] == 1:
                heaviest = max(heaviest, self.list_items[i].weight)
                weight += self.list_items[i].weight
        res["heaviest_item"] = heaviest
        res["weight"] = weight
        return res

    def evaluate_mobj_from_dict(self, dict_values: Dict[str, float]):
        return TupleFitness(
            np.array([dict_values["value"], -dict_values["heaviest_item"]]), 2
        )

    def evaluate_mobj(self, solution: KnapsackSolution):
        return self.evaluate_mobj_from_dict(self.evaluate(solution))


class KnapsackSolutionMultidimensional(Solution):
    value: float
    weights: List[float]
    list_taken: List[bool]

    def __init__(self, problem, list_taken, value=None, weights=None):
        self.problem = problem
        self.value = value
        self.weights = weights
        self.list_taken = list_taken

    def copy(self):
        return KnapsackSolutionMultidimensional(
            problem=self.problem,
            value=self.value,
            weights=self.weights,
            list_taken=list(self.list_taken),
        )

    def lazy_copy(self):
        return KnapsackSolutionMultidimensional(
            problem=self.problem,
            value=self.value,
            weights=self.weights,
            list_taken=self.list_taken,
        )

    def change_problem(self, new_problem):
        self.__init__(
            problem=new_problem,
            value=self.value,
            weights=self.weights,
            list_taken=list(self.list_taken),
        )

    def __str__(self):
        s = "Value=" + str(self.value) + "\n"
        s += "Weights=" + str(self.weights) + "\n"
        s += "Taken : " + str(self.list_taken)
        return s

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return self.list_taken == other.list_taken


class ItemMultidimensional(NamedTuple):
    index: int
    value: float
    weights: List[float]

    def __str__(self):
        return (
            "ind: "
            + str(self.index)
            + " weights: "
            + str(self.weights)
            + " value: "
            + str(self.value)
        )


class MultidimensionalKnapsack(Problem):
    def __init__(
        self,
        list_items: List[ItemMultidimensional],
        max_capacities: List[float],
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
            "weight_violation": {
                "type": TypeObjective.PENALTY,
                "default_weight": -100000,
            },
            "value": {"type": TypeObjective.OBJECTIVE, "default_weight": 1},
        }
        return ObjectiveRegister(
            objective_sense=ModeOptim.MAXIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc=dict_objective,
        )

    def evaluate_from_encoding(self, int_vector, encoding_name):
        if encoding_name == "list_taken":
            kp_sol = KnapsackSolutionMultidimensional(
                problem=self, list_taken=int_vector
            )
        elif encoding_name == "custom":
            kwargs = {encoding_name: int_vector, "problem": self}
            kp_sol = KnapsackSolutionMultidimensional(**kwargs)
        objectives = self.evaluate(kp_sol)
        return objectives

    def evaluate(self, knapsack_solution: KnapsackSolutionMultidimensional):
        if knapsack_solution.value is None or self.force_recompute_values:
            val = self.evaluate_value(knapsack_solution)
        else:
            val = knapsack_solution.value
        w_violation = self.evaluate_weight_violation(knapsack_solution)
        return {"value": val, "weight_violation": w_violation}

    def evaluate_value(self, knapsack_solution: KnapsackSolutionMultidimensional):
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
        self, knapsack_solution: KnapsackSolutionMultidimensional
    ):
        return sum(
            [
                max(0.0, knapsack_solution.weights[j] - self.max_capacities[j])
                for j in range(len(self.max_capacities))
            ]
        )

    def satisfy(self, knapsack_solution: KnapsackSolutionMultidimensional):
        if knapsack_solution.value is None:
            self.evaluate(knapsack_solution)
        return all(
            knapsack_solution.weights[j] <= self.max_capacities[j]
            for j in range(len(self.max_capacities))
        )

    def __str__(self):
        s = (
            "Knapsack model with "
            + str(self.nb_items)
            + " items and capacity "
            + str(self.max_capacities)
            + "\n"
        )
        s += "\n".join([str(item) for item in self.list_items])
        return s

    def get_dummy_solution(self):
        kp_sol = KnapsackSolutionMultidimensional(
            problem=self, list_taken=[0] * self.nb_items
        )
        self.evaluate(kp_sol)
        return kp_sol

    def get_solution_type(self):
        return KnapsackSolutionMultidimensional

    def copy(self):
        return MultidimensionalKnapsack(
            list_items=[deepcopy(x) for x in self.list_items],
            max_capacities=list(self.max_capacities),
            force_recompute_values=self.force_recompute_values,
        )


class MultiScenarioMultidimensionalKnapsack(RobustProblem):
    def __init__(
        self,
        list_problem: List[MultidimensionalKnapsack],
        method_aggregating: MethodAggregating,
    ):
        super().__init__(list_problem, method_aggregating)

    def get_dummy_solution(self):
        return self.list_problem[0].get_dummy_solution()


def from_kp_to_multi(knapsack_model: KnapsackModel):
    return MultidimensionalKnapsack(
        list_items=[
            ItemMultidimensional(index=x.index, value=x.value, weights=[x.weight])
            for x in knapsack_model.list_items
        ],
        max_capacities=[knapsack_model.max_capacity],
    )


def create_noised_scenario(problem: MultidimensionalKnapsack, nb_scenarios: int = 20):
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
