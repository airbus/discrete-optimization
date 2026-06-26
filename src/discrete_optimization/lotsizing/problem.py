#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
# Implementation of the discrete lot sizing problem
# https://www.csplib.org/Problems/prob058/data/
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from discrete_optimization.generic_tools.do_problem import (
    EncodingRegister,
    ModeOptim,
    ObjectiveDoc,
    ObjectiveHandling,
    ObjectiveRegister,
    Problem,
    Solution,
    TypeObjective,
)
from discrete_optimization.generic_tools.encoding_register import (
    ListInteger,
)

logger = logging.getLogger(__name__)


@dataclass
class ProductionItem:
    item_type: int
    quantity: int
    time: int


class LotSizingSolution(Solution):
    problem: LotSizingProblem

    def __init__(
        self,
        problem: LotSizingProblem,
        productions: list[ProductionItem] = None,
        deliveries: list[ProductionItem] = None,
        list_item_per_time: list[int] = None,
    ):
        """

        :param problem:
        :param productions: list of production item.
        :param deliveries: list of delivery item.
        :param list_item_per_time: alternative modeling, working for binary problem,
        for each time, we indicate the item_type to be produced.
        None or nb_item_type or any number not in [0, nb_item_type( corresponds to idle time
        """
        super().__init__(problem)
        self.productions = productions
        self.deliveries = deliveries
        self.recompute = False
        self.list_item_per_time = list_item_per_time
        if productions is None and list_item_per_time is None:
            logger.error(
                f"You need to specify at least a production or item type per time"
            )
        if productions is None and list_item_per_time is not None:
            self.recompute_prod()
        if deliveries is None:
            self.recompute_deliveries()
        if list_item_per_time is None:
            list_item_per_time = [
                self.problem.nb_items_type for i in range(self.problem.horizon)
            ]
            for p in self.productions:
                list_item_per_time[p.time] = p.item_type
            self.list_item_per_time = list_item_per_time
        self.recompute = True

    def copy(self) -> Solution:
        return LotSizingSolution(
            self.problem,
            list(self.productions),
            list(self.deliveries),
            list(self.list_item_per_time),
        )

    def lazy_copy(self) -> Solution:
        return LotSizingSolution(
            self.problem, self.productions, self.deliveries, self.list_item_per_time
        )

    def __setattr__(self, key, value):
        if key == "list_item_per_time":
            super().__setattr__(key, value)
            if value is not None and self.recompute:
                self.recompute_prod()
                self.recompute_deliveries()
            self.recompute = True
        else:
            super().__setattr__(key, value)

    def recompute_prod(self):
        productions = [
            ProductionItem(item_type=self.list_item_per_time[i], quantity=1, time=i)
            for i in range(len(self.list_item_per_time))
            if self.list_item_per_time[i] in self.problem.items_range
        ]
        self.productions = productions

    def recompute_deliveries(self):
        deliveries = []
        prod_per_time = {t: [] for t in range(self.problem.horizon)}
        for prod in self.productions:
            prod_per_time[prod.time].append(prod)
        cur_stock = {item: 0 for item in self.problem.items_range}
        cur_delivery = {item: 0 for item in self.problem.items_range}
        cumul_demands = {
            item: np.cumsum(self.problem.demands[item])
            for item in self.problem.items_range
        }
        for t in range(self.problem.horizon):
            for p in prod_per_time[t]:
                cur_stock[p.item_type] += p.quantity
            for item in cumul_demands:
                if cumul_demands[item][t] > cur_delivery[item]:
                    if cur_stock[item] > 0:
                        deliver_quantity = min(
                            cur_stock[item], cumul_demands[item][t] - cur_delivery[item]
                        )
                        cur_stock[item] -= deliver_quantity
                        cur_delivery[item] += deliver_quantity
                        deliveries.append(
                            ProductionItem(
                                item_type=item, quantity=deliver_quantity, time=t
                            )
                        )
        self.deliveries = deliveries


class LotSizingProblem(Problem):
    def __init__(
        self,
        nb_items_type: int,
        capacity_machine: int,
        changeover_costs: list[list[int]],
        demands: list[list[int]],
        stock_capacity: int,
        stock_cost_per_type_per_time_per_unit: list[int],
        delay_cost_per_type_per_time_per_unit: list[int],
        allow_delays: bool = False,
        known_bound: int = None,
    ):
        self.horizon = len(list(demands[0]))
        self.nb_items_type = nb_items_type
        self.capacity_machine = capacity_machine
        self.changeover_costs = changeover_costs
        self.is_binary = all((y in {0, 1} for x in demands for y in x))
        self.demands = demands
        self.stock_capacity = stock_capacity
        self.stock_cost_per_type_per_time_per_unit = (
            stock_cost_per_type_per_time_per_unit
        )
        self.delay_cost_per_type_per_time_per_unit = (
            delay_cost_per_type_per_time_per_unit
        )
        self.items_range = range(self.nb_items_type)
        self.items_set = set(list(self.items_range))
        self.items_range_with_dummy = range(self.nb_items_type + 1)
        self.allow_delays = allow_delays
        self.total_demands_per_item = {
            item: sum(self.demands[item]) for item in self.items_range
        }
        self.known_bound = known_bound

    def evaluate(self, variable: LotSizingSolution) -> dict[str, float]:
        _, _, stock_cost, delay_cost = compute_stock_and_delays(self, variable)
        changeover_cost = 0
        sorted_prod: list[ProductionItem] = sorted(
            variable.productions, key=lambda p: p.time
        )
        for i in range(len(sorted_prod) - 1):
            changeover_cost += self.changeover_costs[sorted_prod[i].item_type][
                sorted_prod[i + 1].item_type
            ]

        return {
            "delays": float(delay_cost),
            "stock": float(stock_cost),
            "changeover": changeover_cost,
        }

    def satisfy(self, variable: LotSizingSolution) -> bool:
        stock, _, stock_cost, delay_cost = compute_stock_and_delays(self, variable)
        if not self.allow_delays:
            if delay_cost > 0:
                logger.debug(f"Some delays occured, while not allowed")
                return False
        total_stock = stock[0]
        for i in range(1, self.nb_items_type):
            total_stock += stock[i]
        if np.max(total_stock) > self.stock_capacity:
            logger.debug(f"Stock excess")
            return False
        sorted_prod: list[ProductionItem] = sorted(
            variable.productions, key=lambda p: p.time
        )
        times = [p.time for p in sorted_prod]
        if len(set(times)) != len(times):
            logger.debug(f"Times not unique")
            return False
        for p in sorted_prod:
            if p.quantity > self.capacity_machine:
                logger.debug(f"Capacity exceeded")
                return False
        return True

    def get_solution_type(self) -> type[Solution]:
        return LotSizingSolution

    def get_attribute_register(self) -> EncodingRegister:
        return EncodingRegister(
            dict_attribute_to_type={
                "list_item_per_time": ListInteger(
                    length=self.horizon, lows=0, ups=self.nb_items_type
                )
            }
        )

    def get_objective_register(self) -> ObjectiveRegister:
        return ObjectiveRegister(
            objective_sense=ModeOptim.MINIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc={
                "delays": ObjectiveDoc(TypeObjective.PENALTY, default_weight=1),
                "stock": ObjectiveDoc(TypeObjective.PENALTY, default_weight=1),
                "changeover": ObjectiveDoc(TypeObjective.PENALTY, default_weight=1),
            },
        )


def compute_stock_and_delays(problem: LotSizingProblem, solution: LotSizingSolution):
    sorted_prod: list[ProductionItem] = sorted(
        solution.productions, key=lambda p: p.time
    )
    prod_per_time = {t: [] for t in range(problem.horizon)}
    delivery_per_time = {t: [] for t in range(problem.horizon)}
    for prod in solution.productions:
        prod_per_time[prod.time].append(prod)
    for delivery in solution.deliveries:
        delivery_per_time[delivery.time].append(delivery)
    cumul_demand = {
        item_type: np.cumsum(problem.demands[item_type])
        for item_type in range(problem.nb_items_type)
    }
    cumul_prod_per_item_type = {
        item_type: np.zeros(problem.horizon)
        for item_type in range(problem.nb_items_type)
    }
    cumul_delivery_per_item_type = {
        item_type: np.zeros(problem.horizon)
        for item_type in range(problem.nb_items_type)
    }
    stock_per_item_type = {
        item_type: None for item_type in range(problem.nb_items_type)
    }
    delays_per_item_type = {
        item_type: None for item_type in range(problem.nb_items_type)
    }
    for s in sorted_prod:
        item_type = s.item_type
        item_quantity = s.quantity
        cumul_prod_per_item_type[item_type][s.time] += item_quantity
    for d in solution.deliveries:
        item_type = d.item_type
        item_quantity = d.quantity
        cumul_delivery_per_item_type[item_type][d.time] += item_quantity

    for item_type in cumul_prod_per_item_type:
        cumul_prod_per_item_type[item_type] = np.cumsum(
            cumul_prod_per_item_type[item_type]
        )
        cumul_delivery_per_item_type[item_type] = np.cumsum(
            cumul_delivery_per_item_type[item_type]
        )
        stock_per_item_type[item_type] = (
            cumul_prod_per_item_type[item_type]
            - cumul_delivery_per_item_type[item_type]
        )
        delays_per_item_type[item_type] = np.maximum(
            cumul_demand[item_type] - cumul_delivery_per_item_type[item_type], 0
        )
    stock_cost = 0
    delay_cost = 0
    for item_type in stock_per_item_type:
        stock_cost += problem.stock_cost_per_type_per_time_per_unit[item_type] * np.sum(
            stock_per_item_type[item_type]
        )
        delay_cost += problem.delay_cost_per_type_per_time_per_unit[item_type] * np.sum(
            delays_per_item_type[item_type]
        )

    return (stock_per_item_type, delays_per_item_type, stock_cost, delay_cost)
