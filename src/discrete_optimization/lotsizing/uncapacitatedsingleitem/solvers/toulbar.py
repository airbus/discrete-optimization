#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.toulbar_tools import ToulbarSolver
from discrete_optimization.lotsizing.uncapacitatedsingleitem.problem import (
    UncapacitatedSingleItemLSP,
    UncapacitatedSingleItemSolution,
)

try:
    import pytoulbar2

    toulbar_available = True
except ImportError as e:
    toulbar_available = False


class ToulbarUncapacitatedSingleItemSolver(ToulbarSolver, WarmstartMixin):
    problem: UncapacitatedSingleItemLSP

    def init_model(self, **kwargs) -> None:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        horizon = self.problem.horizon
        item_0 = self.problem.items_list[0]
        total_demand = self.problem.get_total_demand(item_0)
        demand_details = [
            int(self.problem.get_demand(item_0, t)) for t in range(horizon)
        ]

        model = pytoulbar2.CFN()
        production_list = [
            model.AddVariable(name=f"prod_{t}", values=range(total_demand))
            for t in range(horizon)
        ]
        inventory_list = [
            model.AddVariable(name=f"inv_{t}", values=range(total_demand))
            for t in range(horizon)
        ]
        for t in range(horizon):
            if t == 0:
                model.AddLinearConstraint(
                    [1, -1],
                    [inventory_list[0], production_list[0]],
                    operand="==",
                    rightcoef=-demand_details[t],
                )
            else:
                # inv[t] = inv[t-1] + prod[t] - demand[t]
                model.AddLinearConstraint(
                    [1, -1, -1],
                    [inventory_list[t], inventory_list[t - 1], production_list[t]],
                    operand="==",
                    rightcoef=-demand_details[t],
                )
        # Production cost :
        for t in range(horizon):
            model.AddFunction(
                [production_list[t]],
                [
                    i * self.problem.get_production_cost_per_unit(item_0, period=t)
                    for i in range(total_demand)
                ],
            )

            # Setup cost, only active when non zero production.
            model.AddFunction(
                [production_list[t]],
                [
                    (i > 0) * self.problem.get_setup_cost(item_0, period=t)
                    for i in range(total_demand)
                ],
            )
            model.AddFunction(
                [inventory_list[t]],
                [
                    i * self.problem.get_inventory_cost_per_unit(item_0, period=t)
                    for i in range(total_demand)
                ],
            )
        self.model = model

    def retrieve_solution(
        self, solution_from_toulbar2: tuple[list, float, int]
    ) -> UncapacitatedSingleItemSolution:
        prod_on_horizon = solution_from_toulbar2[0][: self.problem.horizon]
        production_periods = []
        production_quantities = []
        for i in range(len(prod_on_horizon)):
            if prod_on_horizon[i] > 0:
                production_periods.append(i)
                production_quantities.append(prod_on_horizon[i])
        sol = UncapacitatedSingleItemSolution(
            problem=self.problem,
            production_periods=production_periods,
            production_quantities=production_quantities,
        )
        return sol

    def set_warm_start(self, solution: UncapacitatedSingleItemSolution) -> None:
        item = self.problem.items_list[0]
        for t in range(self.problem.horizon):
            prod = solution.get_production_quantity(item=item, period=t)
            inventory = solution.get_inventory_level(item=item, period=t)
            self.model.CFN.wcsp.setBestValue(t, prod)
            self.model.CFN.wcsp.setBestValue(t + self.problem.horizon, inventory)
