#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.lotsizing.generic_solver.cpsat.lotsing_solver_cpsat import (
    Item,
    LotSizingCpSatSolver,
)


class InventoryConstraintCpsat(LotSizingCpSatSolver[Item]):
    def create_constraint_inventory(self):
        if self.problem.has_stock_limits():
            for t in range(self.problem.horizon):
                if self.problem.has_overall_stock_limit(period=t):
                    overall_limit = self.problem.get_overall_stock_limit(period=t)
                    self.cp_model.add(
                        sum(
                            [
                                self.get_inventory_var(item=item, period=t)
                                for item in self.problem.items_list
                            ]
                        )
                        <= int(overall_limit)
                    )
                for item in self.problem.items_list:
                    if self.problem.has_stock_limit_for_item(item=item, period=t):
                        self.cp_model.add(
                            self.get_inventory_var(item=item, period=t)
                            <= self.problem.get_stock_limit_for_item(
                                item=item, period=t
                            )
                        )
        # Definition of the inventory.
        if not self.problem.is_backlog_allowed():
            for t in range(self.problem.horizon):
                for item in self.problem.items_list:
                    self.cp_model.add(
                        self.get_delivery_var(item=item, period=t)
                        == self.problem.get_demand(item=item, period=t)
                    )
                    if t == 0:
                        self.cp_model.add(
                            self.get_inventory_var(item=item, period=t)
                            == self.get_production_quantity_var(item=item, period=t)
                            - self.problem.get_demand(item=item, period=t)
                        )
                    else:
                        self.cp_model.add(
                            self.get_inventory_var(item=item, period=t)
                            == self.get_inventory_var(item=item, period=t - 1)
                            + self.get_production_quantity_var(item=item, period=t)
                            - self.problem.get_demand(item=item, period=t)
                        )
        else:
            for t in range(self.problem.horizon):
                for item in self.problem.items_list:
                    if t == 0:
                        self.cp_model.add(
                            self.get_inventory_var(item=item, period=t)
                            == self.get_production_quantity_var(item=item, period=t)
                            - self.get_delivery_var(item=item, period=t)
                        )
                    else:
                        self.cp_model.add(
                            self.get_inventory_var(item=item, period=t)
                            == self.get_inventory_var(item=item, period=t - 1)
                            + self.get_production_quantity_var(item=item, period=t)
                            - self.get_delivery_var(item=item, period=t)
                        )

    def create_inventory_cost(self):
        inventory_cost_terms = []
        for item in self.problem.items_list:
            for t in range(self.problem.horizon):
                cost_per_unit = int(self.problem.get_inventory_cost_per_unit(item, t))
                if cost_per_unit > 0:
                    inventory_cost_terms.append(
                        cost_per_unit * self.get_inventory_var(item=item, period=t)
                    )
        return sum(inventory_cost_terms)
