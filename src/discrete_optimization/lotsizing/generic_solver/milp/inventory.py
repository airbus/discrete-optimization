#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Inventory constraints and costs for MILP generic lot sizing solver."""

from discrete_optimization.lotsizing.generic_solver.milp.lotsizing_solver_milp import (
    Item,
    LotSizingMilpSolver,
)


class InventoryConstraintMilp(LotSizingMilpSolver[Item]):
    """Inventory constraints and costs for MILP lot sizing solver.

    This mixin adds:
    - Stock limit constraints (overall and per-item)
    - Inventory balance equations
    - Inventory cost objective terms
    """

    def create_constraint_inventory(self):
        """Create inventory constraints.

        This creates:
        1. Stock limit constraints (if applicable)
        2. Inventory balance equations: I_it = I_i,t-1 + X_it - D_it

        The inventory balance depends on whether backlog is allowed:
        - Without backlog: D_it = d_it (delivery = demand)
        - With backlog: D_it is a variable (delivery may be delayed)
        """
        # Stock limit constraints
        if self.problem.has_stock_limits():
            for t in range(self.problem.horizon):
                # Overall stock limit (sum across all items)
                if self.problem.has_overall_stock_limit(period=t):
                    overall_limit = self.problem.get_overall_stock_limit(period=t)
                    inventory_terms = [
                        self.get_inventory_var(item=item, period=t)
                        for item in self.problem.items_list
                    ]
                    self.add_linear_constraint(
                        self.construct_linear_sum(inventory_terms)
                        <= int(overall_limit),
                        name=f"overall_stock_limit_t{t}",
                    )

                # Per-item stock limits
                for item in self.problem.items_list:
                    if self.problem.has_stock_limit_for_item(item=item, period=t):
                        limit = self.problem.get_stock_limit_for_item(
                            item=item, period=t
                        )
                        self.add_linear_constraint(
                            self.get_inventory_var(item=item, period=t) <= int(limit),
                            name=f"stock_limit_{item}_{t}",
                        )

        # Inventory balance equations
        if not self.problem.is_backlog_allowed():
            # Without backlog: delivery = demand
            for t in range(self.problem.horizon):
                for item in self.problem.items_list:
                    demand = int(self.problem.get_demand(item=item, period=t))

                    # Delivery must equal demand
                    self.add_linear_constraint(
                        self.get_delivery_var(item=item, period=t) == demand,
                        name=f"delivery_equals_demand_{item}_{t}",
                    )

                    # Inventory balance
                    if t == 0:
                        # I_i0 = X_i0 - d_i0
                        self.add_linear_constraint(
                            self.get_inventory_var(item=item, period=t)
                            == self.get_production_quantity_var(item=item, period=t)
                            - demand,
                            name=f"inventory_balance_{item}_{t}",
                        )
                    else:
                        # I_it = I_i,t-1 + X_it - d_it
                        self.add_linear_constraint(
                            self.get_inventory_var(item=item, period=t)
                            == self.get_inventory_var(item=item, period=t - 1)
                            + self.get_production_quantity_var(item=item, period=t)
                            - demand,
                            name=f"inventory_balance_{item}_{t}",
                        )
        else:
            # With backlog: delivery is a variable
            for t in range(self.problem.horizon):
                for item in self.problem.items_list:
                    # Inventory balance: I_it = I_i,t-1 + X_it - D_it
                    if t == 0:
                        # I_i0 = X_i0 - D_i0
                        self.add_linear_constraint(
                            self.get_inventory_var(item=item, period=t)
                            == self.get_production_quantity_var(item=item, period=t)
                            - self.get_delivery_var(item=item, period=t),
                            name=f"inventory_balance_{item}_{t}",
                        )
                    else:
                        # I_it = I_i,t-1 + X_it - D_it
                        self.add_linear_constraint(
                            self.get_inventory_var(item=item, period=t)
                            == self.get_inventory_var(item=item, period=t - 1)
                            + self.get_production_quantity_var(item=item, period=t)
                            - self.get_delivery_var(item=item, period=t),
                            name=f"inventory_balance_{item}_{t}",
                        )

    def create_inventory_cost(self):
        """Create inventory holding cost expression.

        Returns a linear expression for the total inventory holding cost:
            sum_{i,t} (h_it * I_it)

        Where h_it is the inventory holding cost per unit per period.

        Returns:
            Linear expression for inventory cost
        """
        inventory_cost_terms = []
        for item in self.problem.items_list:
            for t in range(self.problem.horizon):
                cost_per_unit = int(self.problem.get_inventory_cost_per_unit(item, t))
                if cost_per_unit > 0:
                    inventory_cost_terms.append(
                        cost_per_unit * self.get_inventory_var(item=item, period=t)
                    )

        if inventory_cost_terms:
            return self.construct_linear_sum(inventory_cost_terms)
        else:
            return 0
