#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Toulbar solvers for capacitated multi-item lot sizing problem."""

from typing import Any

from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.toulbar_tools import ToulbarSolver
from discrete_optimization.lotsizing.capacitatedmultiitem.problem import (
    CapacitatedMultiItemLSP,
    CapacitatedMultiItemSolution,
)

try:
    import pytoulbar2

    toulbar_available = True
except ImportError:
    toulbar_available = False


class ToulbarCapacitatedLotSizingSolver(ToulbarSolver):
    def retrieve_solution(
        self, solution_from_toulbar2: tuple[list, float, int]
    ) -> Solution:
        """Extract solution from Toulbar2 variables.

        Variable order in solution_from_toulbar2[0]:
        1. item_per_time[0..horizon-1]
        2. item_per_time_persistent[0..horizon-1]
        3. cumulative_prod[item][0..horizon-1] for each item
        """
        # Create solution using productions
        sol = CapacitatedMultiItemSolution(
            problem=self.problem,
            list_item_per_time=solution_from_toulbar2[0][: self.problem.horizon],
        )

        print(f"\n=== Solution evaluation ===")
        print(f"problem.evaluate(sol): {self.problem.evaluate(sol)}")
        print(f"problem.satisfy(sol): {self.problem.satisfy(sol)}")
        print(f"Toulbar cost: {solution_from_toulbar2[1]}")

        return sol

    problem: CapacitatedMultiItemLSP

    def init_model(self, **kwargs: Any) -> None:
        model = pytoulbar2.CFN()
        # Get objective weights from params_objective_function
        obj_weights = {}
        for obj, weight in zip(
            self.params_objective_function.objectives,
            self.params_objective_function.weights,
        ):
            obj_weights[obj] = weight

        changeover_weight = obj_weights.get("changeover_cost", 1.0)
        backlog_weight = obj_weights.get("backlog_cost", 1.0)
        inv_weight = obj_weights.get("inventory_cost", 1.0)

        item_per_time = [
            model.AddVariable(
                name=f"prod_time_{t}", values=range(self.problem.nb_items + 1)
            )
            for t in range(self.problem.horizon)
        ]
        # For changeover...
        item_per_time_persistent = [
            model.AddVariable(
                name=f"last_item_{t}", values=range(self.problem.nb_items + 1)
            )
            for t in range(self.problem.horizon)
        ]
        model.AddLinearConstraint(
            [1, -1],
            [item_per_time[0], item_per_time_persistent[0]],
            operand="==",
            rightcoef=0,
        )
        model.AddGeneralizedLinearConstraint(
            [
                [item_per_time[t], self.problem.nb_items, 1]
                for t in range(self.problem.horizon)
            ],
            operand="==",
            rightcoef=self.problem.horizon
            - sum(
                [
                    self.problem.get_total_demand(item)
                    for item in range(self.problem.nb_items)
                ]
            ),
        )
        for t in range(1, self.problem.horizon):
            flat_cost = []
            tuple_list = []
            tuple_cost = []
            for i in range(self.problem.nb_items + 1):
                for j in range(self.problem.nb_items + 1):
                    for k in range(self.problem.nb_items + 1):
                        if k == self.problem.nb_items:
                            if i != j:
                                # when k is idle time, we want to force
                                # item_per_time_persistent[t] == item_per_time_persistent[t-1]
                                flat_cost.append(model.Top)
                            else:
                                flat_cost.append(0)
                                tuple_list.append([i, j, k])
                                tuple_cost.append(0)
                        if k != self.problem.nb_items:
                            if i == k:
                                # changeover cost
                                if j == self.problem.nb_items:
                                    flat_cost.append(0)
                                    tuple_list.append([i, j, k])
                                    tuple_cost.append(0)
                                else:
                                    tuple_list.append([i, j, k])
                                    tuple_cost.append(
                                        int(
                                            self.problem.get_changeover_cost(
                                                from_item=j, to_item=i
                                            )
                                        )
                                    )
                                    # flat_cost.append(int(self.problem.get_changeover_cost(from_item=j,
                                    #                                                      to_item=i)))
                                    flat_cost.append(0)
                            else:
                                # when item_per_time is not idle,
                                # we want to sync the 2...
                                flat_cost.append(model.Top)

            model.AddFunction(
                [
                    item_per_time_persistent[t],
                    item_per_time_persistent[t - 1],
                    item_per_time[t],
                ],
                flat_cost,
            )
            model.AddFunction(
                [item_per_time_persistent[t], item_per_time_persistent[t - 1]],
                [
                    self.problem.get_changeover_cost(j, i)
                    if i != self.problem.nb_items and j != self.problem.nb_items
                    else 0
                    for i in range(self.problem.nb_items + 1)
                    for j in range(self.problem.nb_items + 1)
                ],
            )
            # model.AddCompactFunction([item_per_time_persistent[t],
            #                    item_per_time_persistent[t-1],
            #                    item_per_time[t]],
            #                          defcost=model.Top,
            #                          tuples=tuple_list,
            #                          tcosts=tuple_cost)
        cumulative_prod = {}
        for item in self.problem.items_list:
            cumulative_prod[item] = [
                model.AddVariable(
                    name=f"cumulative_prod_{item}_{t}",
                    values=range(self.problem.get_total_demand(item) + 1),
                )
                for t in range(self.problem.horizon)
            ]
            model.AddFunction(
                [cumulative_prod[item][0], item_per_time[0]],
                costs=[
                    0 if (j == item and i == 1) or (j != item and i == 0) else model.Top
                    for i in range(self.problem.get_total_demand(item) + 1)
                    for j in range(self.problem.nb_items + 1)
                ],
            )
            # model.AddLinearConstraint([1], [cumulative_prod[item][0]],
            #                          operand="<=", rightcoef=1)
            for t in range(1, self.problem.horizon):
                # model.AddLinearConstraint([1, -1],
                #                            [cumulative_prod[item][t],
                #                             cumulative_prod[item][t - 1]],
                #                            operand=">=",
                #                            rightcoef=0)
                model.AddLinearConstraint(
                    [1, -1],
                    [cumulative_prod[item][t], cumulative_prod[item][t - 1]],
                    operand="<=",
                    rightcoef=1,
                )
                tuples_list = []
                tuples_cost = []
                flat_cost = []
                for new_val in range(self.problem.get_total_demand(item) + 1):
                    for prev_val in range(self.problem.get_total_demand(item) + 1):
                        for item_ in range(self.problem.nb_items + 1):
                            if new_val == prev_val + 1 and item_ == item:
                                tuples_list.append([new_val, prev_val, item_])
                                tuples_cost.append(0)
                                flat_cost.append(0)
                            elif new_val == prev_val and item_ != item:
                                tuples_list.append([new_val, prev_val, item_])
                                tuples_cost.append(0)
                                flat_cost.append(0)
                            else:
                                flat_cost.append(model.Top)

                model.AddCompactFunction(
                    scope=[
                        cumulative_prod[item][t],
                        cumulative_prod[item][t - 1],
                        item_per_time[t],
                    ],
                    defcost=1e7,
                    tuples=tuples_list,
                    tcosts=tuples_cost,
                )
                model.AddFunction(
                    scope=[
                        cumulative_prod[item][t],
                        cumulative_prod[item][t - 1],
                        item_per_time[t],
                    ],
                    costs=flat_cost,
                )
                # model.AddGeneralizedLinearConstraint([[cumulative_prod[item][t], val, val]
                #                                       for val in range(self.problem.get_total_demand(item)+1)]
                #                                      +[[item_per_time[tprime], item, -1] for tprime in range(t+1)],
                #                                      operand="==",
                #                                      rightcoef=0)
                model.AddGeneralizedLinearConstraint(
                    [
                        [cumulative_prod[item][t], val, val]
                        for val in range(self.problem.get_total_demand(item) + 1)
                    ]
                    + [
                        [cumulative_prod[item][t - 1], val, -val]
                        for val in range(self.problem.get_total_demand(item) + 1)
                    ]
                    + [[item_per_time[t], item, -1]],
                    operand="==",
                    rightcoef=0,
                )

        # Objective: backlog and inventory costs
        for item in self.problem.items_list:
            cumul = self.problem.get_cumulative_demands(item)
            total_demand = int(self.problem.get_total_demand(item))

            for t in range(self.problem.horizon):
                costs = []
                for i in range(total_demand + 1):
                    if i < cumul[t]:
                        # Backlog: demand not yet met
                        cost = int(
                            self.problem.get_backlog_cost_per_unit(item, period=t)
                            * backlog_weight
                            * (cumul[t] - i)
                        )
                        cost = model.Top
                    else:
                        # Inventory: produced more than demanded
                        cost = int(
                            self.problem.get_inventory_cost_per_unit(item, period=t)
                            * inv_weight
                            * (i - cumul[t])
                        )
                    costs.append(cost)

                model.AddFunction([cumulative_prod[item][t]], costs=costs)
            model.AddGeneralizedLinearConstraint(
                [[item_per_time[t], item, 1] for t in range(self.problem.horizon)],
                operand="==",
                rightcoef=self.problem.get_total_demand(item),
            )
            model.AddLinearConstraint(
                [1],
                [cumulative_prod[item][-1]],
                "==",
                int(self.problem.get_total_demand(item)),
            )
        self.model = model
        self.variables = {}
        self.variables["item_per_time"] = item_per_time
