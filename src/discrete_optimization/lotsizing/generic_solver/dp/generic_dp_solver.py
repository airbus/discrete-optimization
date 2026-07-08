#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from typing import Any, Optional

import didppy as dp

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
)
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.dyn_prog_tools import DpSolver
from discrete_optimization.lotsizing.generic_lotsizing import (
    GenericLotSizingProblem,
)
from discrete_optimization.lotsizing.production_solution import (
    ProductionBasedSolution,
    ProductionDecision,
)

logger = logging.getLogger(__name__)


class GenericLotSizingDp(DpSolver, WarmstartMixin):
    allow_backorder: bool = False
    max_backorder: int = 5
    force_unmet_zero: bool = True
    add_transition_dominance: bool = False
    penalty_advance_time: int = 500000
    lookahead_demand: int = 5
    add_additional_dual_bounds: bool = False
    problem: GenericLotSizingProblem

    def __init__(
        self,
        problem: GenericLotSizingProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs,
    ):
        super().__init__(problem, params_objective_function, **kwargs)

    def init_model(self, **kwargs: Any) -> None:
        self.model = dp.Model()
        cumul_demands = {
            item: self.problem.get_cumulative_demands(item)
            for item in self.problem.items_list
        }
        demands_with_lookahead = {
            item: [
                cumul_demands[item][
                    min(t + self.lookahead_demand, self.problem.horizon - 1)
                ]
                - cumul_demands[item][t]
                for t in range(self.problem.horizon)
            ]
            + [0]
            for item in self.problem.items_list
        }

        item_with_dummy = self.model.add_object_type(number=self.problem.nb_items + 1)
        setup_times = [
            [
                int(self.problem.get_setup_time(item=item, period=t))
                for t in range(self.problem.horizon)
            ]
            for item in self.problem.items_list
        ]
        setup_costs = [
            [
                int(self.problem.get_setup_cost(item=item, period=t))
                for t in range(self.problem.horizon)
            ]
            for item in self.problem.items_list
        ]
        usage = [
            [
                int(self.problem.get_production_time_per_unit(item, t))
                for t in range(self.problem.horizon)
            ]
            for item in self.problem.items_list
        ]
        dummy_capa = 1000
        if self.problem.has_capacity_limits():
            dummy_capa = int(self.problem.get_available_production_time(period=0))
        current_capacity = self.model.add_int_var(target=dummy_capa)
        time_object = self.model.add_object_type(number=self.problem.horizon + 1)
        current_time = self.model.add_int_var(target=0)
        current_time_element = self.model.add_element_var(
            object_type=time_object, target=0
        )
        stock = [self.model.add_int_var(target=0) for item in self.problem.items_list]
        cumul_prod = [
            self.model.add_int_var(target=0) for item in self.problem.items_list
        ]
        demands = [
            self.model.add_int_table(
                [
                    int(self.problem.get_demand(item, t))
                    for t in range(self.problem.horizon)
                ]
            )
            for item in self.problem.items_list
        ]
        self.cumulated_demands = [
            self.model.add_int_table(cumul_demands[item])
            for item in self.problem.items_list
        ]
        self.cumulated_demands_look = [
            self.model.add_int_table(demands_with_lookahead[item])
            for item in self.problem.items_list
        ]
        self.cumulated_deliveries = [
            self.model.add_int_state_fun(cumul_prod[item] - stock[item])
            for item in self.problem.items_list
        ]
        self.setup_costs = [
            self.model.add_int_table(setup_costs[i])
            for i in range(len(self.problem.items_list))
        ]
        self.setup_times = [
            self.model.add_int_table(setup_times[i])
            for i in range(len(self.problem.items_list))
        ]
        self.usage_time = [
            self.model.add_int_table(usage[i])
            for i in range(len(self.problem.items_list))
        ]
        if self.problem.has_capacity_limits():
            self.capacity_calendar = self.model.add_int_table(
                [
                    int(self.problem.get_available_production_time(t))
                    for t in range(self.problem.horizon)
                ]
                + [0]
            )
        else:
            self.capacity_calendar = None
        self.cumul_prod = cumul_prod
        self.demands = demands
        self.current_item = self.model.add_element_var(
            object_type=item_with_dummy, target=self.problem.nb_items
        )
        self.changeover_matrix = self.model.add_int_table(
            [
                [
                    int(self.problem.get_changeover_cost(from_item=i, to_item=j))
                    for j in range(self.problem.nb_items)
                ]
                + [0]
                for i in range(len(self.problem.items_list))
            ]
            + [[0 for j in range(self.problem.nb_items + 1)]]
        )
        self.current_time = current_time
        self.current_time_element = current_time_element
        self.stock = stock
        self.future_demand = [
            self.model.add_int_state_fun(
                self.cumulated_demands_look[i][self.current_time_element]
                - self.stock[i]
            )
            for i in range(self.problem.nb_items)
        ]

        self.future_prod_to_do = [
            self.model.add_int_state_fun(
                dp.max(
                    self.cumulated_demands_look[i][self.current_time_element]
                    - self.stock[i],
                    0,
                )
            )
            for i in range(self.problem.nb_items)
        ]
        future = dp.IntExpr(0)
        for i in range(self.problem.nb_items):
            future += self.future_demand[i]
        if not self.problem.allows_parallel_production():
            m = 0
            for item in self.problem.items_list:
                i = self.problem.get_index_from_item(item)
                max_prod = self.problem.get_max_production_quantity(item, period=0)
                m = max(max_prod, m)
                self.model.add_state_constr(
                    self.future_prod_to_do[i] <= max_prod * self.lookahead_demand
                )
            self.model.add_state_constr(
                sum(self.future_prod_to_do) <= m * self.lookahead_demand
            )

        self.current_capacity = current_capacity
        self.transitions = self.transition_schedule()

        # Add multiple informative dual bounds for better guidance
        self.model.add_dual_bound(0)
        if self.add_additional_dual_bounds:
            self.add_dual_bounds()
        # for item in self.problem.items_list:
        #    self.model.add_state_constr(self.cumul_prod[item]<=int(self.problem.get_total_demand(item)))
        if self.force_unmet_zero:
            self.model.add_base_case(
                [self.current_time_element == self.problem.horizon]
            )
        else:
            self.model.add_base_case(
                [self.current_time_element == self.problem.horizon]
            )

    def add_dual_bounds(self) -> None:
        """Add multiple dual bounds to guide the search.

        Each bound provides a different perspective on the remaining cost,
        giving the solver more information for pruning.
        """
        # Precompute minimum costs
        min_setup_costs = []
        min_inventory_costs = []
        min_changeover_costs = []
        avg_inventory_costs = []

        for item in self.problem.items_list:
            min_setup = min(
                int(self.problem.get_setup_cost(item, t))
                for t in range(self.problem.horizon)
            )
            min_setup_costs.append(min_setup)

            min_inv = min(
                int(self.problem.get_inventory_cost_per_unit(item, t))
                for t in range(self.problem.horizon)
            )
            min_inventory_costs.append(min_inv)

            avg_inv = (
                sum(
                    int(self.problem.get_inventory_cost_per_unit(item, t))
                    for t in range(self.problem.horizon)
                )
                // self.problem.horizon
            )
            avg_inventory_costs.append(avg_inv)

            min_co = min(
                int(self.problem.get_changeover_cost(from_item=from_item, to_item=item))
                for from_item in self.problem.items_list
            )
            min_changeover_costs.append(min_co)

        total_demands = [
            int(self.problem.get_total_demand(item)) for item in self.problem.items_list
        ]

        # Bound 1: Setup costs only (very optimistic)
        bound1 = dp.IntExpr(0)
        for i, item in enumerate(self.problem.items_list):
            remaining_demand = total_demands[i] - self.cumul_prod[i]
            has_remaining = dp.max(0, dp.min(remaining_demand, 1))
            bound1 += has_remaining * min_setup_costs[i]
        self.model.add_dual_bound(bound1)

        # Bound 2: Setup + one minimal changeover (assumes single switch)
        bound2 = dp.IntExpr(0)
        num_items_remaining = dp.IntExpr(0)
        for i, item in enumerate(self.problem.items_list):
            remaining_demand = total_demands[i] - self.cumul_prod[i]
            has_remaining = dp.max(0, dp.min(remaining_demand, 1))
            num_items_remaining += has_remaining
            bound2 += has_remaining * min_setup_costs[i]
        min_co_overall = min(min_changeover_costs) if min_changeover_costs else 0
        has_multiple = dp.max(0, dp.min(num_items_remaining - 1, 1))
        bound2 += has_multiple * min_co_overall
        self.model.add_dual_bound(bound2)

        # Bound 3: Setup + full changeover per item (no inventory)
        bound3 = dp.IntExpr(0)
        for i, item in enumerate(self.problem.items_list):
            remaining_demand = total_demands[i] - self.cumul_prod[i]
            has_remaining = dp.max(0, dp.min(remaining_demand, 1))
            bound3 += has_remaining * (min_setup_costs[i] + min_changeover_costs[i])
        self.model.add_dual_bound(bound3)

        # Bound 4: Setup + changeover + minimal inventory
        bound4 = dp.IntExpr(0)
        for i, item in enumerate(self.problem.items_list):
            remaining_demand = total_demands[i] - self.cumul_prod[i]
            has_remaining = dp.max(0, dp.min(remaining_demand, 1))
            bound4 += has_remaining * (min_setup_costs[i] + min_changeover_costs[i])

            # Current stock holding cost (at least 1 period)
            remaining_periods = dp.max(0, self.problem.horizon - self.current_time)
            periods_to_hold = dp.min(remaining_periods, 1)
            bound4 += self.stock[i] * min_inventory_costs[i] * periods_to_hold
        self.model.add_dual_bound(bound4)

        # Bound 5: Aggressive inventory bound (assumes stock must be held longer)
        bound5 = dp.IntExpr(0)
        for i, item in enumerate(self.problem.items_list):
            remaining_demand = total_demands[i] - self.cumul_prod[i]
            has_remaining = dp.max(0, dp.min(remaining_demand, 1))
            bound5 += has_remaining * min_setup_costs[i]

            # Assume stock is held for multiple periods (more aggressive)
            remaining_periods = dp.max(0, self.problem.horizon - self.current_time)
            periods_to_hold = dp.min(
                remaining_periods, 2
            )  # At least 2 periods if possible
            bound5 += self.stock[i] * min_inventory_costs[i] * periods_to_hold
        self.model.add_dual_bound(bound5)

        # Bound 6: Average inventory cost bound (between min and max)
        bound6 = dp.IntExpr(0)
        for i, item in enumerate(self.problem.items_list):
            remaining_demand = total_demands[i] - self.cumul_prod[i]
            has_remaining = dp.max(0, dp.min(remaining_demand, 1))
            bound6 += has_remaining * min_setup_costs[i]
            bound6 += has_remaining * min_changeover_costs[i]

            remaining_periods = dp.max(0, self.problem.horizon - self.current_time)
            periods_to_hold = dp.min(remaining_periods, 1)
            bound6 += self.stock[i] * avg_inventory_costs[i] * periods_to_hold
        self.model.add_dual_bound(bound6)

        # Bound 7: Heuristic bound based on remaining demand volume
        # Items with more remaining demand likely need more setups (optimistic estimate)
        bound7 = dp.IntExpr(0)
        for i, item in enumerate(self.problem.items_list):
            remaining_demand = total_demands[i] - self.cumul_prod[i]
            # Assume we need at least ceil(remaining/max_batch) setups
            # Approximated as: if remaining > 0, at least min_setup_cost
            has_remaining = dp.max(0, dp.min(remaining_demand, 1))
            bound7 += has_remaining * min_setup_costs[i]

            # Add changeover from current item specifically (more realistic)
            # This is slightly optimistic but guides towards the current item context
            bound7 += has_remaining * min_changeover_costs[i]
        self.model.add_dual_bound(bound7)

        # Bound 8: Pessimistic changeover (heuristic using avg of min/max)
        max_co_costs = [
            max(
                int(self.problem.get_changeover_cost(from_item=f, to_item=item))
                for f in self.problem.items_list
            )
            for item in self.problem.items_list
        ]
        bound8 = dp.IntExpr(0)
        for i, item in enumerate(self.problem.items_list):
            remaining = total_demands[i] - self.cumul_prod[i]
            has_rem = dp.max(0, dp.min(remaining, 1))
            heur_co = (min_changeover_costs[i] + max_co_costs[i]) // 3
            bound8 += has_rem * (min_setup_costs[i] + heur_co)
        self.model.add_dual_bound(bound8)

        # Bound 9: Stock penalty (penalizes holding stock)
        bound9 = dp.IntExpr(0)
        total_stock = sum(self.stock[i] for i in range(len(self.problem.items_list)))
        for i in range(len(self.problem.items_list)):
            remaining = total_demands[i] - self.cumul_prod[i]
            has_rem = dp.max(0, dp.min(remaining, 1))
            bound9 += has_rem * min_setup_costs[i]
        bound9 += total_stock * (min(min_inventory_costs) if min_inventory_costs else 0)
        self.model.add_dual_bound(bound9)

        # Bound 10: Average changeover heuristic
        avg_co = (
            sum(min_changeover_costs) // len(min_changeover_costs)
            if min_changeover_costs
            else 0
        )
        bound10 = dp.IntExpr(0)
        for i in range(len(self.problem.items_list)):
            remaining = total_demands[i] - self.cumul_prod[i]
            has_rem = dp.max(0, dp.min(remaining, 1))
            bound10 += has_rem * (min_setup_costs[i] + avg_co // 2)
        self.model.add_dual_bound(bound10)

        # Bound 11: Stock holding with time pressure
        bound11 = dp.IntExpr(0)
        for i in range(len(self.problem.items_list)):
            remaining = total_demands[i] - self.cumul_prod[i]
            has_rem = dp.max(0, dp.min(remaining, 1))
            bound11 += has_rem * min_setup_costs[i]
            bound11 += self.stock[i] * min_inventory_costs[i]
        self.model.add_dual_bound(bound11)

    def transition_schedule(self):
        transitions = {}
        for item in self.problem.items_list:
            max_produce = 0
            for t in range(self.problem.horizon):
                max_produce = max(
                    self.problem.get_max_production_quantity(item, t), max_produce
                )
            max_produce = min(max_produce, self.problem.get_total_demand(item))
            for prod in range(1, max_produce + 1):
                effects = []
                preconditions = []
                preconditions.append(
                    self.cumul_prod[item] + prod
                    <= int(self.problem.get_total_demand(item))
                )
                cost = dp.IntExpr(0)
                if not self.problem.allows_parallel_production():
                    preconditions.append(
                        self.current_time_element <= self.problem.horizon - 1
                    )
                    cost += self.changeover_matrix[self.current_item, item]
                    # we advance in time by scheduling
                    effects.append(
                        (self.current_time_element, self.current_time_element + 1)
                    )
                    effects.append((self.current_time, self.current_time + 1))
                    effects.append((self.current_item, item))
                    for item_ in self.problem.items_list:
                        stock_virtual = self.stock[item_]
                        prod_virtual = self.cumul_prod[item_]
                        if item_ == item:
                            stock_virtual += prod
                            prod_virtual += prod
                        deliver = dp.min(
                            stock_virtual,
                            self.cumulated_demands[item_][self.current_time_element]
                            - self.cumulated_deliveries[item_],
                        )
                        backlog = self.cumulated_demands[item_][
                            self.current_time_element
                        ] - (self.cumulated_deliveries[item_] + deliver)
                        if not self.allow_backorder:
                            preconditions.append(
                                prod_virtual
                                >= self.cumulated_demands[item_][
                                    self.current_time_element
                                ]
                            )
                        stock = stock_virtual - deliver
                        cost += backlog * int(
                            self.problem.get_backlog_cost_per_unit(item_, 0)
                        )
                        cost += stock * int(
                            self.problem.get_inventory_cost_per_unit(item_, 0)
                        )

                        if item_ == item:
                            effects.append(
                                (self.cumul_prod[item_], self.cumul_prod[item_] + prod)
                            )
                            cost += self.setup_costs[item][self.current_time_element]
                        if self.max_backorder is not None:
                            preconditions.append(
                                prod_virtual
                                >= self.cumulated_demands[item_][
                                    self.current_time_element
                                ]
                                - self.max_backorder
                            )

                        effects.append((self.stock[item_], stock_virtual - deliver))
                    if self.capacity_calendar is not None:
                        effects.append(
                            (
                                self.current_capacity,
                                self.capacity_calendar[self.current_time_element + 1],
                            )
                        )
                else:
                    effects.append((self.current_item, item))
                    for item_ in self.problem.items_list:
                        if item_ == item:
                            effects.append(
                                (self.cumul_prod[item_], self.cumul_prod[item_] + prod)
                            )
                            effects.append(
                                (self.stock[item_], self.stock[item_] + prod)
                            )
                            if self.problem.has_capacity_limits():
                                effects.append(
                                    (
                                        self.current_capacity,
                                        self.current_capacity
                                        - self.setup_times[item_][
                                            self.current_time_element
                                        ]
                                        - prod
                                        * self.usage_time[item_][
                                            self.current_time_element
                                        ],
                                    )
                                )
                            cost += self.setup_costs[item][self.current_time_element]
                            if not self.allow_backorder:
                                preconditions.append(
                                    self.cumul_prod[item_] + prod
                                    >= self.cumulated_demands[item_][
                                        self.current_time_element
                                    ]
                                )
                    cost += self.changeover_matrix[self.current_item, item]
                if self.problem.has_capacity_limits():
                    preconditions.append(
                        self.current_capacity
                        >= self.setup_times[item][self.current_time_element]
                        + self.usage_time[item][self.current_time_element] * prod
                    )
                preconditions.append(
                    self.current_time_element <= self.problem.horizon - 1
                )
                tr = dp.Transition(
                    name=f"prod_{prod}_{item}",
                    cost=cost + dp.IntExpr.state_cost(),
                    effects=effects,
                    preconditions=preconditions,
                )
                id = self.model.add_transition(tr)
                transitions[f"prod_{prod}_{item}"] = {
                    "tr": tr,
                    "id": id,
                    "meaning": ("prod", prod, item),
                }
        if self.add_transition_dominance:
            for item1 in self.problem.items_list:
                for item2 in self.problem.items_list:
                    if item1 == item2:
                        continue
                    id1 = transitions[f"prod_{1}_{item1}"]["id"]
                    id2 = transitions[f"prod_{1}_{item2}"]["id"]
                    self.model.add_transition_dominance(
                        id1,
                        id2,
                        conditions=[
                            self.cumulated_demands[item1][self.current_time_element]
                            - self.cumul_prod[item1]
                            > 0,
                            self.cumulated_demands[item1][self.current_time_element]
                            - self.cumul_prod[item1]
                            > self.cumulated_demands[item2][self.current_time_element]
                            - self.cumul_prod[item2],
                        ],
                    )
                    self.model.add_transition_dominance(
                        id1,
                        id2,
                        conditions=[
                            self.stock[item1]
                            < self.demands[item1][self.current_time_element],
                            self.stock[item2]
                            >= self.demands[item2][self.current_time_element],
                        ],
                    )
                    # self.model.add_transition_dominance(id1, id2,
                    #                                    conditions=[self.stock[item2]>=self.stock[item1],
                    #                                                self.changeover_matrix[self.current_item, item1]<
                    #                                                self.changeover_matrix[self.current_item, item2]])
                    self.model.add_transition_dominance(
                        id1,
                        id2,
                        conditions=[
                            self.cumul_prod[item1]
                            < self.cumulated_demands[item1][self.current_time_element],
                            self.cumul_prod[item2]
                            >= self.cumulated_demands[item2][self.current_time_element],
                        ],
                    )
        # Advance time :
        effects = []
        effects.append((self.current_time_element, self.current_time_element + 1))
        effects.append((self.current_time, self.current_time + 1))
        # effects.append((self.count_idle, self.count_idle+1))
        cost = dp.IntExpr(0)
        preconditions = []
        # preconditions = [self.count_idle<=
        #                 self.problem.horizon-total_demand]
        for item_ in self.problem.items_list:
            stock = self.stock[item_]
            # prod = self.cumul_prod[item_]
            deliver = dp.min(
                stock,
                self.cumulated_demands[item_][self.current_time_element]
                - self.cumulated_deliveries[item_],
            )
            backlog = -(
                self.cumulated_deliveries[item_]
                + deliver
                - self.cumulated_demands[item_][self.current_time_element]
            )
            if not self.problem.is_backlog_allowed():
                preconditions.append(
                    self.cumul_prod[item_]
                    >= self.cumulated_demands[item_][self.current_time_element]
                )
            if self.max_backorder is not None:
                preconditions.append(
                    self.cumul_prod[item_]
                    >= self.cumulated_demands[item_][self.current_time_element]
                    - self.max_backorder
                )
            cost += backlog * int(self.problem.get_backlog_cost_per_unit(item_, 0))
            cost += (stock - deliver) * int(
                self.problem.get_inventory_cost_per_unit(item_, 0)
            )
            if int(self.problem.get_backlog_cost_per_unit(item_, 0)) == 0:
                cost += backlog * 10000
            effects.append((self.stock[item_], self.stock[item_] - deliver))
        if self.capacity_calendar is not None:
            effects.append(
                (
                    self.current_capacity,
                    self.capacity_calendar[self.current_time_element + 1],
                )
            )

        preconditions.append((self.current_time_element <= self.problem.horizon - 1))
        advance_time = dp.Transition(
            name="advance_time",
            cost=self.penalty_advance_time + cost + dp.IntExpr.state_cost(),
            effects=effects,
            preconditions=preconditions,
        )
        id2 = self.model.add_transition(advance_time)
        transitions[f"advance_time"] = {
            "tr": advance_time,
            "id2": id2,
            "meaning": ("advance_time", None, None),
        }
        for item in self.problem.items_list:
            id1 = transitions[f"prod_{1}_{item}"]["id"]
            self.model.add_transition_dominance(
                id1,
                id2,
                conditions=[
                    self.current_time_element < self.problem.horizon - 1,
                    self.cumul_prod[item]
                    < self.cumulated_demands[item][self.current_time_element],
                ],
            )
        # Advance ti

        return transitions

    def retrieve_solution(self, sol: dp.Solution) -> ProductionBasedSolution:
        state = self.model.target_state
        time = 0
        productions = []
        for t in sol.transitions:
            # print("cost of transition", t.name,
            #      t.eval_cost(0, state, self.model))
            state = t.apply(state, self.model)
            name = t.name
            meaning = self.transitions[name]["meaning"]
            if meaning[0] == "prod":
                prod, item = meaning[1], meaning[2]
                productions.append(
                    ProductionDecision(item=item, period=time, quantity=prod)
                )

            time = state[self.current_time_element]

        return ProductionBasedSolution(problem=self.problem, productions=productions)

    def set_warm_start(self, solution: ProductionBasedSolution) -> None:
        pass
