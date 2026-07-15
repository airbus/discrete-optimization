#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Production constraints and costs for MILP generic lot sizing solver."""

from discrete_optimization.lotsizing.generic_solver.milp.lotsizing_solver_milp import (
    Item,
    LotSizingMilpSolver,
)


class ProductionConstraintMilp(LotSizingMilpSolver[Item]):
    """Production constraints and costs for MILP lot sizing solver.

    This mixin adds:
    - Capacity constraints (production time + setup time <= available time)
    - Setup cost objective terms
    - Production cost objective terms
    """

    def create_constraint_production(self):
        """Create production capacity constraints.

        For each period t, ensures that total production time and setup time
        does not exceed the available production time:
            sum_i (X_it * p_it + Y_it * s_it) <= C_t

        Where:
        - X_it: production quantity of item i in period t
        - p_it: production time per unit for item i in period t
        - Y_it: setup binary variable (1 if production occurs)
        - s_it: setup time for item i in period t
        - C_t: available production time in period t
        """
        if self.problem.has_capacity_limits():
            for t in range(self.problem.horizon):
                available_time = int(
                    self.problem.get_available_production_time(period=t)
                )
                if available_time < float("inf"):
                    # Build capacity constraint terms
                    capacity_terms = []
                    for item in self.problem.items_list:
                        # Production time
                        prod_time = int(
                            self.problem.get_production_time_per_unit(
                                item=item, period=t
                            )
                        )
                        if prod_time > 0:
                            capacity_terms.append(
                                prod_time
                                * self.get_production_quantity_var(item=item, period=t)
                            )

                        # Setup time
                        setup_time = int(
                            self.problem.get_setup_time(item=item, period=t)
                        )
                        if setup_time > 0:
                            capacity_terms.append(
                                setup_time
                                * self.get_production_binary_var(item=item, period=t)
                            )

                    if capacity_terms:
                        self.add_linear_constraint(
                            self.construct_linear_sum(capacity_terms) <= available_time,
                            name=f"capacity_constraint_t{t}",
                        )

                    # Redundant constraints: individual production limits
                    for item in self.problem.items_list:
                        max_prod = self.problem.get_max_production_quantity(
                            item=item, period=t
                        )
                        if max_prod < float("inf"):
                            self.add_linear_constraint(
                                self.get_production_quantity_var(item=item, period=t)
                                <= max_prod,
                                name=f"max_production_{item}_{t}",
                            )

    def create_production_cost(self):
        """Create production cost expression.

        Returns a linear expression for the total production cost:
            sum_{i,t} (v_it * X_it)

        Where v_it is the production cost per unit.

        Returns:
            Linear expression for production cost
        """
        production_cost_terms = []
        for item in self.problem.items_list:
            for t in range(self.problem.horizon):
                cost_per_unit = int(self.problem.get_production_cost_per_unit(item, t))
                if cost_per_unit > 0:
                    production_cost_terms.append(
                        cost_per_unit
                        * self.get_production_quantity_var(item=item, period=t)
                    )

        if production_cost_terms:
            return self.construct_linear_sum(production_cost_terms)
        else:
            return 0

    def create_setup_cost(self):
        """Create setup cost expression.

        Returns a linear expression for the total setup cost:
            sum_{i,t} (s_it * Y_it)

        Where s_it is the fixed setup cost.

        Returns:
            Linear expression for setup cost
        """
        setup_cost_terms = []
        for item in self.problem.items_list:
            for t in range(self.problem.horizon):
                cost = int(self.problem.get_setup_cost(item, t))
                if cost > 0:
                    setup_cost_terms.append(
                        cost * self.get_production_binary_var(item=item, period=t)
                    )

        if setup_cost_terms:
            return self.construct_linear_sum(setup_cost_terms)
        else:
            return 0
