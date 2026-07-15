#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Wagner-Whitin dynamic programming solver for uncapacitated single-item lot sizing.

This module implements the classical Wagner-Whitin algorithm (1958) that solves
the uncapacitated single-item lot sizing problem optimally in O(T²) time.

References:
    Wagner, H. M., & Whitin, T. M. (1958).
    Dynamic version of the economic lot size model.
    Management science, 5(1), 89-96.
"""

from __future__ import annotations

import logging

import numpy as np

from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.lotsizing.production_solution import ProductionDecision
from discrete_optimization.lotsizing.uncapacitatedsingleitem.problem import (
    UncapacitatedSingleItemLSP,
    UncapacitatedSingleItemSolution,
)

logger = logging.getLogger(__name__)


class WagnerWhitinSolver(SolverDO):
    """Wagner-Whitin dynamic programming solver.

    This solver computes the optimal solution for the uncapacitated single-item
    lot sizing problem in O(T²) time using dynamic programming.

    The algorithm exploits the Zero Inventory Ordering (ZIO) property:
    In an optimal solution, production in period t (if any) should satisfy
    demand for an integer number of consecutive future periods, with zero
    inventory before period t.

    This means we only need to consider T² possible solutions (for each period,
    consider producing to satisfy demand up to each future period).

    The dynamic programming recurrence is:
        f[t] = min over k in [0, t) of [f[k] + cost(k, t)]

    Where:
        f[t] = minimum cost to satisfy demands from period 0 to t-1
        cost(k, t) = cost to produce in period k to satisfy demands k to t-1
                   = setup_cost[k] + production_cost[k] * sum(demands[k:t])
                     + sum of inventory costs for holding from k to t

    Base case: f[0] = 0 (no cost before first period)

    Time complexity: O(T²) where T is the horizon
    Space complexity: O(T)
    """

    problem: UncapacitatedSingleItemLSP

    def __init__(self, problem: UncapacitatedSingleItemLSP, **kwargs):
        """Initialize Wagner-Whitin solver.

        Args:
            problem: Uncapacitated single-item lot sizing problem instance
            **kwargs: Additional solver parameters (ignored)
        """
        super().__init__(problem, **kwargs)

    def solve(self, **kwargs) -> ResultStorage:
        """Solve the problem optimally using Wagner-Whitin algorithm.

        Args:
            **kwargs: Additional parameters (ignored)

        Returns:
            ResultStorage containing the optimal solution
        """
        logger.info("Running Wagner-Whitin DP algorithm...")

        T = self.problem.horizon
        item = 0  # Single item

        # Precompute cumulative demands for efficiency
        demands = [self.problem.get_demand(item, t) for t in range(T)]
        cumul_demands = np.cumsum([0] + demands)  # cumul_demands[t] = sum(demands[0:t])

        # DP table: f[t] = minimum cost to satisfy demands from 0 to t-1
        f = [float("inf")] * (T + 1)
        f[0] = 0.0  # Base case: no cost before first period

        # Backtracking: best_k[t] = best production period for subproblem ending at t
        best_k = [-1] * (T + 1)

        # Dynamic programming: compute f[t] for t = 1, 2, ..., T
        for t in range(1, T + 1):
            # Try all possible last production periods k in [0, t)
            for k in range(t):
                # Cost to produce in period k to satisfy demands from k to t-1
                cost_k_t = self._compute_production_cost(k, t, cumul_demands)

                # Total cost: cost up to k + cost to produce k to t
                total_cost = f[k] + cost_k_t

                # Update if better
                if total_cost < f[t]:
                    f[t] = total_cost
                    best_k[t] = k

        # Backtrack to find optimal production plan
        production_decisions = []
        t = T
        while t > 0:
            k = best_k[t]
            if k == -1:
                raise RuntimeError(f"No valid production plan found for period {t}")

            # Produce in period k to satisfy demands from k to t-1
            quantity = int(cumul_demands[t] - cumul_demands[k])
            if quantity > 0:
                production_decisions.append(
                    ProductionDecision(item=0, period=k, quantity=quantity)
                )

            t = k

        # Reverse to get chronological order
        production_decisions.reverse()

        # Create solution
        solution = UncapacitatedSingleItemSolution(
            problem=self.problem,
            production_periods=[p.period for p in production_decisions],
            production_quantities=[p.quantity for p in production_decisions],
        )

        # Verify solution
        optimal_cost = f[T]
        computed_cost = solution.compute_total_cost()

        logger.info(f"Optimal cost: {optimal_cost:.2f}")
        logger.info(f"Computed cost: {computed_cost:.2f}")
        logger.info(f"Production periods: {[p.period for p in production_decisions]}")

        if abs(optimal_cost - computed_cost) > 1e-6:
            logger.warning(
                f"Cost mismatch: DP says {optimal_cost:.2f}, "
                f"solution says {computed_cost:.2f}"
            )

        # Create result storage
        result_storage = ResultStorage(
            mode_optim=self.problem.get_objective_register().objective_sense,
            list_solution_fits=[
                (solution, self.aggreg_from_dict(self.problem.evaluate(solution)))
            ],
        )

        return result_storage

    def _compute_production_cost(
        self,
        k: int,
        t: int,
        cumul_demands: np.ndarray,
    ) -> float:
        """Compute cost to produce in period k to satisfy demands from k to t-1.

        Args:
            k: Production period
            t: End period (exclusive)
            cumul_demands: Cumulative demands array

        Returns:
            Total cost (setup + production + inventory)
        """
        item = 0

        # Setup cost in period k
        setup_cost = self.problem.get_setup_cost(item, k)

        # Production quantity: sum of demands from k to t-1
        quantity = cumul_demands[t] - cumul_demands[k]

        # Production cost
        production_cost_per_unit = self.problem.get_production_cost_per_unit(item, k)
        production_cost = production_cost_per_unit * quantity

        # Inventory cost: for each period p in [k, t-1], we hold inventory
        # Inventory at end of period p = demands still to be delivered = sum(demands[p+1:t])
        # which is: cumul_demands[t] - cumul_demands[p+1]
        inventory_cost = 0.0
        for p in range(k, t):
            # Inventory at end of period p (demands p+1 through t-1 not yet delivered)
            inv = cumul_demands[t] - cumul_demands[p + 1]
            inv_cost_per_unit = self.problem.get_inventory_cost_per_unit(item, p)
            inventory_cost += inv * inv_cost_per_unit

        total_cost = setup_cost + production_cost + inventory_cost

        return total_cost
