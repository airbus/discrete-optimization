#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.lotsizing.generic_solver.cpsat.lotsizing_solver_cpsat import (
    Item,
    LotSizingCpSatSolver,
)


class ProductionConstraintCpsat(LotSizingCpSatSolver[Item]):
    def create_constraint_production(self):
        # Cumul - kind of constraint
        if self.problem.has_capacity_limits():
            for t in range(self.problem.horizon):
                available_time = int(
                    self.problem.get_available_production_time(period=t)
                )
                if available_time < float("inf"):
                    self.cp_model.add(
                        sum(
                            [
                                self.get_production_quantity_var(item=item, period=t)
                                * int(
                                    self.problem.get_production_time_per_unit(
                                        item=item, period=t
                                    )
                                )  # Production time
                                + self.get_production_binary_var(item=item, period=t)
                                * int(
                                    self.problem.get_setup_time(item=item, period=t)
                                )  # Setup time
                                for item in self.problem.items_list
                            ]
                        )
                        <= available_time
                    )
                    # redundant =
                    for item in self.problem.items_list:
                        self.cp_model.add(
                            self.get_production_quantity_var(item=item, period=t)
                            <= self.problem.get_max_production_quantity(
                                item=item, period=t
                            )
                        )

    def create_production_cost(self):
        production_cost_terms = []
        for item in self.problem.items_list:
            for t in range(self.problem.horizon):
                cost_per_unit = int(self.problem.get_production_cost_per_unit(item, t))
                if cost_per_unit > 0:
                    production_cost_terms.append(
                        cost_per_unit
                        * self.get_production_quantity_var(item=item, period=t)
                    )
        return sum(production_cost_terms)

    def create_setup_cost(self):
        setup_cost_terms = []
        for item in self.problem.items_list:
            for t in range(self.problem.horizon):
                cost_per_unit = int(self.problem.get_setup_cost(item, t))
                if cost_per_unit > 0:
                    setup_cost_terms.append(
                        cost_per_unit
                        * self.get_production_binary_var(item=item, period=t)
                    )
        return sum(setup_cost_terms)
