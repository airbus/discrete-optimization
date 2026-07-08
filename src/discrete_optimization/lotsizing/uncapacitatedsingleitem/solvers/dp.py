#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any

import didppy as dp

from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.dyn_prog_tools import DpSolver
from discrete_optimization.lotsizing.uncapacitatedsingleitem.problem import (
    UncapacitatedSingleItemLSP,
    UncapacitatedSingleItemSolution,
)


class DpUncapacitatedLotSizingSolver(DpSolver, WarmstartMixin):
    problem: UncapacitatedSingleItemLSP
    transition_name: dict
    transition_objects: dict

    def init_model(self, **kwargs: Any) -> None:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        horizon = self.problem.horizon
        item_0 = self.problem.items_list[0]
        model = dp.Model()
        total_demand = self.problem.get_total_demand(item_0)
        cumulative_prod = model.add_int_var(target=0)
        current_stock = model.add_int_var(target=0)
        # cumulative_demand = [int(cd) for cd in self.problem.get_cumulative_demands(item_0)]
        time_object = model.add_object_type(self.problem.horizon + 1)
        current_time = model.add_element_var(object_type=time_object, target=0)
        model.add_state_constr(cumulative_prod <= total_demand)
        cost_product = model.add_int_table(
            [
                int(self.problem.get_production_cost_per_unit(item_0, period=t))
                for t in range(self.problem.horizon)
            ]
        )
        setup_cost = model.add_int_table(
            [
                int(self.problem.get_setup_cost(item_0, period=t))
                for t in range(self.problem.horizon)
            ]
        )
        demand_table = model.add_int_table(
            [
                int(self.problem.get_demand(item_0, period=t))
                for t in range(self.problem.horizon)
            ]
        )
        inventory_cost = model.add_int_table(
            [
                int(self.problem.get_inventory_cost_per_unit(item_0, period=t))
                for t in range(self.problem.horizon)
            ]
        )
        next_stock_after_produce = [
            model.add_int_state_fun(
                current_stock + quantity - demand_table[current_time]
            )
            for quantity in range(total_demand + 1)
        ]
        self.transition_name = {}
        self.transition_objects = {}

        for quantity in range(1, total_demand + 1):
            trans_name = f"produce_{quantity}"
            tr = dp.Transition(
                name=trans_name,
                cost=quantity * cost_product[current_time]
                + setup_cost[current_time]
                + inventory_cost[current_time] * next_stock_after_produce[quantity]
                + dp.IntExpr.state_cost(),
                effects=[
                    (cumulative_prod, cumulative_prod + quantity),
                    (current_stock, next_stock_after_produce[quantity]),
                    (current_time, current_time + 1),
                ],
                preconditions=[
                    current_time <= horizon - 1,
                    cumulative_prod + quantity <= total_demand,
                    next_stock_after_produce[quantity] >= 0,
                ],
            )
            tr_id = model.add_transition(tr)
            self.transition_name[trans_name] = ("prod", quantity)
            self.transition_objects[trans_name] = tr
        next_time = dp.Transition(
            name="advance",
            cost=inventory_cost[current_time] * next_stock_after_produce[0]
            + dp.IntExpr.state_cost(),
            effects=[
                (current_time, current_time + 1),
                (current_stock, next_stock_after_produce[0]),
            ],
            preconditions=[
                current_time <= horizon - 1,
                next_stock_after_produce[0] >= 0,
            ],
        )
        model.add_transition(next_time)
        model.add_base_case([current_time == horizon])
        self.transition_name["advance"] = ("prod", 0)
        self.transition_objects["advance"] = next_time
        self.model = model

    def retrieve_solution(self, sol: dp.Solution) -> Solution:
        prod_periods = []
        prod_quantities = []
        current_time = 0
        for tr in sol.transitions:
            key, prod = self.transition_name[tr.name]
            if prod > 0:
                prod_periods.append(current_time)
                prod_quantities.append(prod)
            current_time += 1
        return UncapacitatedSingleItemSolution(
            problem=self.problem,
            production_periods=prod_periods,
            production_quantities=prod_quantities,
        )

    def set_warm_start(self, solution: UncapacitatedSingleItemSolution) -> None:
        self.initial_solution = []
        prod = solution.get_production_quantity_array(self.problem.items_list[0])
        for t in range(self.problem.horizon):
            if prod[t] == 0:
                self.initial_solution.append(self.transition_objects["advance"])
            else:
                self.initial_solution.append(
                    self.transition_objects[f"produce_{int(prod[t])}"]
                )
