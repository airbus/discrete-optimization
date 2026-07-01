#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  Homebase for Single item lot sizing problem
#  This should cover capacitated and uncapacitated variants of the problems
#  Slide 9 :
#  https://roadef.org/jfro/static/files/slides_313.pdf
from discrete_optimization.generic_tools.do_problem import (
    ObjectiveRegister,
    Problem,
    Solution,
)


class SingleItemLotSizingProblem(Problem):
    def __init__(
        self,
        dynamic_demand: list[int],
        dynamic_production_cost: list[int],
        dynamic_setup_cost: list[int],
        dynamic_inventory_cost: list[int],
        dynamic_production_time_per_unit: list[int],
        dynamic_available_time_for_production: list[int],
        dynamic_setup_time: list[int],
    ):
        self.dynamic_demand = dynamic_demand
        self.dynamic_production_cost = dynamic_production_cost
        self.dynamic_setup_cost = dynamic_setup_cost
        self.dynamic_inventory_cost = dynamic_inventory_cost
        self.dynamic_production_time_per_unit = dynamic_production_time_per_unit
        self.dynamic_available_time_for_production = (
            dynamic_available_time_for_production
        )
        self.dynamic_setup_time = dynamic_setup_time

    def evaluate(self, variable: Solution) -> dict[str, float]:
        pass

    def satisfy(self, variable: Solution) -> bool:
        pass

    def get_solution_type(self) -> type[Solution]:
        pass

    def get_objective_register(self) -> ObjectiveRegister:
        pass
