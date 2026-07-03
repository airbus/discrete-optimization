#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any

from ortools.sat.python.cp_model import CpSolverSolutionCallback, IntVar

from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver
from discrete_optimization.lotsizing.generic_lotsizing import (
    GenericLotSizingProblem,
    Item,
)
from discrete_optimization.lotsizing.generic_solver.cpsat.lotsing_solver_cpsat import (
    LotSizingCpSatSolver,
)


class GenericLotSizingCpsat(LotSizingCpSatSolver[Item], OrtoolsCpSatSolver):
    problem: GenericLotSizingProblem[Item]
    variables: dict
    production: dict[Item, list[IntVar]]
    inventory: dict[Item, list[IntVar]]
    delivery: dict[Item, list[IntVar]]
    backlog: dict[Item, list[IntVar]]

    def init_model(self, **kwargs: Any) -> None:
        super().init_model(**kwargs)
        self.create_production_vars()
        self.create_inventory_vars()
        self.create_delivery_vars()
        self.create_backlog_vars()

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> Solution:
        pass

    def get_production_var(self, item: Item, period: int) -> Any:
        return self.production[item][period]

    def get_inventory_var(self, item: Item, period: int) -> Any:
        return self.inventory[item][period]

    def get_backlog_var(self, item: Item, period: int) -> Any:
        return self.backlog[item][period]

    def create_production_vars(self):
        for item in self.problem.items_list:
            total_demand = self.problem.get_total_demand(item)
            self.production[item] = [
                self.cp_model.new_int_var(
                    lb=0, ub=total_demand, name=f"production_{item}_{period}"
                )
                for period in range(self.problem.horizon)
            ]

    def create_inventory_vars(self):
        for item in self.problem.items_list:
            total_demand = self.problem.get_total_demand(item)
            self.inventory[item] = [
                self.cp_model.new_int_var(
                    lb=0, ub=total_demand, name=f"inventory_{item}_{period}"
                )
                for period in range(self.problem.horizon)
            ]

    def create_delivery_vars(self):
        for item in self.problem.items_list:
            total_demand = self.problem.get_total_demand(item)
            self.delivery[item] = [
                self.cp_model.new_int_var(
                    lb=0, ub=total_demand, name=f"delivery_{item}_{period}"
                )
                for period in range(self.problem.horizon)
            ]

    def create_backlog_vars(self):
        for item in self.problem.items_list:
            total_demand = self.problem.get_total_demand(item)
            self.backlog[item] = [
                self.cp_model.new_int_var(
                    lb=0, ub=total_demand, name=f"backlog_{item}_{period}"
                )
                for period in range(self.problem.horizon)
            ]
