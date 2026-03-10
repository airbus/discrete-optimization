#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Union

from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveDoc,
    ObjectiveHandling,
    ObjectiveRegister,
    TypeObjective,
)
from discrete_optimization.vrp.problem import (
    BasicCustomer,
    VrpProblem,
    VrpSolution,
    length,
)


class CustomerTop(BasicCustomer):
    def __init__(self, name: Union[str, int], reward: float):
        super().__init__(name, demand=0)
        self.reward = reward


class TeamOrienteeringProblem(VrpProblem):
    customers: list[CustomerTop]
    max_length_tours: float

    def __init__(
        self,
        vehicle_count: int,
        max_length_tours: float,
        customer_count: int,
        customers: Sequence[BasicCustomer],
        start_indexes: list[int],
        end_indexes: list[int],
    ):
        super().__init__(
            vehicle_count,
            [float("inf")] * vehicle_count,
            customer_count,
            customers,
            start_indexes,
            end_indexes,
        )
        self.max_length_tours = max_length_tours

    def evaluate(self, solution: VrpSolution) -> dict[str, float]:
        vals_per_vehicle = self.evaluate_function(solution)
        penalty_length = sum(
            [max(0, x[1] - self.max_length_tours) for x in vals_per_vehicle]
        )
        reward = sum(x[0] for x in vals_per_vehicle)
        count_multiple = self.count_multiple_visits(solution)
        return {
            "reward": reward,
            "penalty_length": penalty_length,
            "penalty_multiple": count_multiple,
        }

    def evaluate_function(
        self, vrp_sol: VrpSolution
    ) -> tuple[list[list[float]], list[float], float, list[float]]:
        vals_per_vehicle = []
        for k in range(self.vehicle_count):
            path = vrp_sol.list_paths[k]
            start = vrp_sol.list_start_index[k]
            end = vrp_sol.list_end_index[k]
            full_path = [start] + path + [end]
            length = sum(
                self.evaluate_function_indexes(xi, xi1)
                for xi, xi1 in zip(full_path[:-1], full_path[1:])
            )
            reward = sum(self.customers[xi].reward for xi in path)
            vals_per_vehicle.append((reward, length))
        return vals_per_vehicle

    def satisfy(self, variable: VrpSolution) -> bool:
        kpis = self.evaluate(variable)
        return kpis["penalty_length"] == 0 and kpis["penalty_multiple"] == 0

    def count_multiple_visits(self, vrp_sol: VrpSolution) -> bool:
        count = defaultdict(lambda: 0)
        for k in range(len(vrp_sol.list_paths)):
            for node in vrp_sol.list_paths[k]:
                count[node] += 1
        return len([n for n in count if count[n] > 1])

    def get_objective_register(self) -> ObjectiveRegister:
        return ObjectiveRegister(
            objective_sense=ModeOptim.MINIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc={
                "reward": ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=1),
                "penalty_length": ObjectiveDoc(
                    type=TypeObjective.PENALTY, default_weight=-100
                ),
                "penalty_multiple": ObjectiveDoc(
                    type=TypeObjective.PENALTY, default_weight=-100
                ),
            },
        )


class CustomerTop2D(CustomerTop):
    def __init__(self, name: Union[str, int], reward: float, x: float, y: float):
        super().__init__(name, reward)
        self.x = x
        self.y = y


class TeamOrienteeringProblem2D(TeamOrienteeringProblem):
    customers: list[CustomerTop]
    max_length_tours: float

    def evaluate_function_indexes(self, index_1: int, index_2: int) -> float:
        return length(self.customers[index_1], self.customers[index_2])
