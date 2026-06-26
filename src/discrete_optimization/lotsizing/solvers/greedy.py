#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from enum import Enum
from typing import Any, Optional

import numpy as np

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    EnumHyperparameter,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.lotsizing.problem import (
    LotSizingProblem,
    LotSizingSolution,
)


class GreedyStrategy(Enum):
    """Greedy strategy for lot sizing problem.

    - EARLIEST_DEMAND_FIRST: Produce items to satisfy earliest unmet demands
    - MINIMUM_CHANGEOVER: Batch production of same items to minimize changeover costs
    - JUST_IN_TIME: Produce items as late as possible while meeting demands
    - BALANCED: Balance between changeover costs and stock holding costs
    """

    EARLIEST_DEMAND_FIRST = "earliest_demand_first"
    MINIMUM_CHANGEOVER = "minimum_changeover"
    JUST_IN_TIME = "just_in_time"
    BALANCED = "balanced"


class GreedyLotSizingSolver(SolverDO):
    """Greedy solver for lot sizing problems with multiple strategies.

    This solver provides various greedy heuristics for constructing initial solutions
    to the lot sizing problem. These solutions can be used as starting points for
    local search or other metaheuristic methods.

    Hyperparameters:
        strategy: The greedy strategy to use for constructing the solution

    Example:
        solver = GreedyLotSizingSolver(problem)
        result = solver.solve(strategy=GreedyStrategy.BALANCED)
    """

    problem: LotSizingProblem

    hyperparameters = [
        EnumHyperparameter(
            name="strategy",
            enum=GreedyStrategy,
            default=GreedyStrategy.BALANCED,
        ),
    ]

    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        strategy = kwargs["strategy"]

        callback = CallbackList(callbacks=callbacks)
        callback.on_solve_start(self)

        if strategy == GreedyStrategy.EARLIEST_DEMAND_FIRST:
            sol = self._earliest_demand_first()
        elif strategy == GreedyStrategy.MINIMUM_CHANGEOVER:
            sol = self._minimum_changeover()
        elif strategy == GreedyStrategy.JUST_IN_TIME:
            sol = self._just_in_time()
        elif strategy == GreedyStrategy.BALANCED:
            sol = self._balanced()
        else:
            sol = self._earliest_demand_first()

        fit = self.aggreg_from_sol(sol)
        res = self.create_result_storage([(sol, fit)])
        callback.on_solve_end(res, self)
        return res

    def _earliest_demand_first(self) -> LotSizingSolution:
        """Produce items to satisfy earliest unmet demands.

        This strategy prioritizes meeting demands as soon as they occur,
        minimizing delay penalties but potentially increasing stock costs.
        """
        list_item_per_time = [self.problem.nb_items_type] * self.problem.horizon

        # Calculate cumulative demand for each item
        cumul_demand = {
            item: np.cumsum(self.problem.demands[item])
            for item in self.problem.items_range
        }

        # Track cumulative production for each item
        cumul_prod = {item: 0.0 for item in self.problem.items_range}

        # Calculate total demand
        total_demand = {
            item: sum(self.problem.demands[item]) for item in self.problem.items_range
        }

        for t in range(self.problem.horizon):
            best_item = None
            best_score = -float("inf")

            for item in self.problem.items_range:
                # Skip if we've already produced enough for this item
                if cumul_prod[item] >= total_demand[item]:
                    continue

                # Calculate how far behind we are
                current_deficit = cumul_demand[item][t] - cumul_prod[item]

                # Calculate future deficit if we don't produce now
                remaining_demand = total_demand[item] - cumul_prod[item]

                # Prioritize by current deficit first, then by remaining demand
                score = current_deficit * 10 + remaining_demand

                if score > best_score:
                    best_score = score
                    best_item = item

            if best_item is not None:
                list_item_per_time[t] = best_item
                cumul_prod[best_item] += self.problem.capacity_machine

        return LotSizingSolution(
            problem=self.problem, list_item_per_time=list_item_per_time
        )

    def _minimum_changeover(self) -> LotSizingSolution:
        """Batch production of same items to minimize changeover costs.

        This strategy produces large batches of each item type to minimize
        the number of changeovers, while respecting demand timing.
        """
        list_item_per_time = [self.problem.nb_items_type] * self.problem.horizon

        # Calculate total demand per item
        total_demand = {
            item: sum(self.problem.demands[item]) for item in self.problem.items_range
        }

        # Calculate required production time for each item
        required_time = {
            item: int(np.ceil(total_demand[item] / self.problem.capacity_machine))
            for item in self.problem.items_range
        }

        # Find first demand time for each item
        first_demand_time = {}
        for item in self.problem.items_range:
            for t in range(self.problem.horizon):
                if self.problem.demands[item][t] > 0:
                    first_demand_time[item] = t
                    break
            if item not in first_demand_time:
                first_demand_time[item] = self.problem.horizon

        # Sort items by first demand time, then by total demand (descending)
        sorted_items = sorted(
            self.problem.items_range,
            key=lambda i: (first_demand_time[i], -total_demand[i]),
        )

        # Allocate production time for each item in batches
        t = 0
        for item in sorted_items:
            periods_needed = required_time[item]
            for _ in range(periods_needed):
                if t < self.problem.horizon:
                    list_item_per_time[t] = item
                    t += 1

        return LotSizingSolution(
            problem=self.problem, list_item_per_time=list_item_per_time
        )

    def _just_in_time(self) -> LotSizingSolution:
        """Produce items as late as possible while meeting demands.

        This strategy minimizes stock holding costs by producing items
        just before they are needed, potentially increasing changeover costs.
        """
        list_item_per_time = [self.problem.nb_items_type] * self.problem.horizon

        # Calculate total demand per item
        total_demand = {
            item: sum(self.problem.demands[item]) for item in self.problem.items_range
        }

        # Track remaining demand to produce
        remaining_to_produce = {
            item: total_demand[item] for item in self.problem.items_range
        }

        # Work backwards from the last period
        for t in range(self.problem.horizon - 1, -1, -1):
            # Find item with most remaining demand to produce
            best_item = None
            max_remaining = 0

            for item in self.problem.items_range:
                if remaining_to_produce[item] > max_remaining:
                    max_remaining = remaining_to_produce[item]
                    best_item = item

            if best_item is not None and max_remaining > 0:
                list_item_per_time[t] = best_item
                remaining_to_produce[best_item] -= min(
                    self.problem.capacity_machine, remaining_to_produce[best_item]
                )

        return LotSizingSolution(
            problem=self.problem, list_item_per_time=list_item_per_time
        )

    def _balanced(self) -> LotSizingSolution:
        """Balance between changeover costs and stock holding costs.

        This strategy considers both the urgency of demands and the cost
        of changeovers to make production decisions.
        """
        list_item_per_time = [self.problem.nb_items_type] * self.problem.horizon

        # Calculate total demand and remaining demand for each item
        total_demand = {
            item: sum(self.problem.demands[item]) for item in self.problem.items_range
        }
        remaining_demand = {
            item: total_demand[item] for item in self.problem.items_range
        }

        # Calculate stock cost rate for each item
        stock_cost_rate = {
            item: self.problem.stock_cost_per_type_per_time_per_unit[item]
            for item in self.problem.items_range
        }

        last_item = None

        for t in range(self.problem.horizon):
            best_item = None
            best_score = -float("inf")

            # Calculate remaining capacity in future periods
            remaining_periods = self.problem.horizon - t

            for item in self.problem.items_range:
                # Skip if no remaining demand
                if remaining_demand[item] <= 0:
                    continue

                # Calculate urgency: how many periods left to produce this item
                required_periods = int(
                    np.ceil(remaining_demand[item] / self.problem.capacity_machine)
                )

                # If we're running out of time, prioritize this item
                urgency = (
                    max(0, (required_periods - remaining_periods) * 100)
                    + remaining_demand[item]
                )

                # Calculate changeover penalty
                changeover_penalty = (
                    0
                    if last_item is None or last_item == item
                    else self.problem.changeover_costs[last_item][item]
                )

                # Calculate stock cost if we produce now vs later
                periods_until_demand = 0
                for future_t in range(t, self.problem.horizon):
                    if self.problem.demands[item][future_t] > 0:
                        break
                    periods_until_demand += 1

                stock_penalty = stock_cost_rate[item] * periods_until_demand

                # Combined score: urgency - penalties
                score = urgency - changeover_penalty - stock_penalty

                if score > best_score:
                    best_score = score
                    best_item = item

            if best_item is not None:
                list_item_per_time[t] = best_item
                remaining_demand[best_item] -= min(
                    self.problem.capacity_machine, remaining_demand[best_item]
                )
                last_item = best_item
            else:
                # If no item needed, keep idle
                last_item = None

        return LotSizingSolution(
            problem=self.problem, list_item_per_time=list_item_per_time
        )


def greedy_best(problem: LotSizingProblem) -> LotSizingSolution:
    """Try all greedy strategies and return the best solution.

    Args:
        problem: The lot sizing problem instance

    Returns:
        The best solution found among all strategies
    """
    best_sol = None
    best_fit = float("inf")

    for strategy in GreedyStrategy:
        solver = GreedyLotSizingSolver(problem)
        result = solver.solve(strategy=strategy)
        sol, fit = result[0]

        if fit < best_fit:
            best_fit = fit
            best_sol = sol

    return best_sol
