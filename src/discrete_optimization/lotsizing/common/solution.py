#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#  Common utils for lot sizing problem.

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from discrete_optimization.generic_tools.do_problem import Problem, Solution

logger = logging.getLogger(__name__)


@dataclass
class ProductionItem:
    item_type: int
    quantity: int
    time: int


class CommonLotSizingSolution(Solution):
    problem: Problem

    def __init__(
        self,
        problem: Problem,
        productions: list[ProductionItem] = None,
        deliveries: list[ProductionItem] = None,
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
        if productions is None:
            logger.error(f"You need to specify at least a production")
        if deliveries is None:
            self.recompute_deliveries()
        self.recompute = True

    def copy(self) -> Solution:
        return CommonLotSizingSolution(
            self.problem,
            list(self.productions),
            list(self.deliveries),
        )

    def lazy_copy(self) -> Solution:
        return CommonLotSizingSolution(
            self.problem,
            self.productions,
            self.deliveries,
        )

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
