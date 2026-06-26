#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Simulated Annealing solver for lot sizing problem.

Implementation based on:
Ceschia, Di Gaspero, Schaerf (2017) - "Solving discrete lot-sizing and
scheduling by simulated annealing and mixed integer programming"
"""

import logging
import math
import random
from enum import Enum
from typing import Any, List, Optional, Tuple

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    FloatHyperparameter,
    IntegerHyperparameter,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.lotsizing.problem import (
    LotSizingProblem,
    LotSizingSolution,
    ProductionItem,
)

logger = logging.getLogger(__name__)


class MoveType(Enum):
    """Types of neighborhood moves."""

    INSERT = "insert"
    SWAP = "swap"


class SimulatedAnnealingLotSizingSolver(SolverDO):
    """Simulated Annealing solver for lot sizing problem.

    Solution representation: Vector V of size horizon
    - V[t] = i means produce item i in period t
    - V[t] = -1 means idle period (no production)

    Neighborhood:
    - Insert: move element from position i to position j
    - Swap: swap elements at positions i and j

    Smart mutations (optional):
    - Stock-aware INSERT: move productions closer to deadlines to reduce stock costs
    - Changeover-aware SWAP: create batching to reduce changeover costs
    - Deadline-aware INSERT: aggressively move productions near deadlines

    Cost function:
    - StockingCost: holding inventory
    - SetupCost: changeover costs
    - NoBacklog penalty: for late deliveries (soft constraint)
    """

    problem: LotSizingProblem

    hyperparameters = [
        FloatHyperparameter(name="T0", low=1.0, high=100.0, default=37.0),
        FloatHyperparameter(name="alpha", low=0.8, high=0.99, default=0.99),
        FloatHyperparameter(name="beta", low=0.0, high=1.0, default=0.7),
        IntegerHyperparameter(name="n_a", low=100, high=10000, default=12049),
        IntegerHyperparameter(name="n_s", low=10, high=1000, default=60240),
        IntegerHyperparameter(
            name="max_iterations", low=1000, high=1000000, default=310000
        ),
        CategoricalHyperparameter(
            name="use_smart_mutations", choices=[True, False], default=False
        ),
        FloatHyperparameter(name="smart_mutation_prob", low=0.0, high=1.0, default=0.3),
    ]

    def __init__(
        self,
        problem: LotSizingProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.T0 = kwargs.get("T0", 37.0)  # Initial temperature
        self.alpha = kwargs.get("alpha", 0.99)  # Cooling rate
        self.beta = kwargs.get("beta", 0.7)  # Insert move probability
        self.n_a = kwargs.get("n_a", 12049)  # Moves accepted at each temperature
        self.n_s = kwargs.get("n_s", 60240)  # Moves sampled at each temperature
        self.max_iterations = kwargs.get("max_iterations", 310000)

        # Smart mutation parameters (optional feature)
        self.use_smart_mutations = kwargs.get("use_smart_mutations", False)
        self.smart_mutation_prob = kwargs.get(
            "smart_mutation_prob", 0.3
        )  # Probability of using smart move when enabled

    def _generate_initial_solution(self) -> List[int]:
        """Generate random initial solution.

        Returns:
            List of length horizon where V[t] is the item produced at time t
            or -1 for idle period.
        """
        # Collect all demands
        all_demands = []
        for item in self.problem.items_range:
            for t in range(self.problem.horizon):
                demand = self.problem.demands[item][t]
                if demand > 0:
                    # Add demand occurrences (assume binary: demand is 0 or 1)
                    for _ in range(int(demand)):
                        all_demands.append((item, t))

        # Create solution vector
        solution = [-1] * self.problem.horizon

        # Randomly assign each demand to a period (respecting NoBacklog later)
        random.shuffle(all_demands)

        # Simple strategy: produce each item in a random period <= deadline
        used_slots = set()
        for item, deadline in all_demands:
            # Find a free slot before or at deadline
            possible_times = [t for t in range(deadline + 1) if t not in used_slots]

            if possible_times:
                t = random.choice(possible_times)
            else:
                # If no free slot before deadline, find first free slot anywhere
                free_slots = [
                    t for t in range(self.problem.horizon) if t not in used_slots
                ]
                if free_slots:
                    t = random.choice(free_slots)
                else:
                    # All slots taken - this shouldn't happen if horizon >= total demands
                    # Fall back to any slot (will overwrite, but better than losing the item)
                    t = random.randint(0, self.problem.horizon - 1)

            solution[t] = item
            used_slots.add(t)

        return solution

    def _evaluate_solution(self, solution: List[int]) -> float:
        """Evaluate cost of a solution.

        Args:
            solution: Production sequence

        Returns:
            Total cost (stocking + setup + NoBacklog penalty)
        """
        horizon = self.problem.horizon
        items = list(self.problem.items_range)

        # Track inventory for each item
        inventory = {i: 0 for i in items}

        # Track last produced item (for changeover cost)
        last_item = None

        # Costs
        stocking_cost = 0.0
        setup_cost = 0.0
        nobacklog_violations = 0

        # Track remaining demands
        remaining_demand = {}
        for item in items:
            for t in range(horizon):
                if self.problem.demands[item][t] > 0:
                    remaining_demand[item, t] = self.problem.demands[item][t]

        # Process each period
        for t in range(horizon):
            produced_item = solution[t]

            # Production
            if produced_item >= 0:
                inventory[produced_item] += 1

                # Setup/changeover cost
                # No cost for initial setup (matches CP-SAT scheduler with dummy node)
                if last_item is not None and last_item != produced_item:
                    setup_cost += self.problem.changeover_costs[last_item][
                        produced_item
                    ]

                last_item = produced_item

            # Satisfy demands and calculate stocking cost
            for item in items:
                demand_at_t = self.problem.demands[item][t]

                if demand_at_t > 0:
                    if inventory[item] >= demand_at_t:
                        # Satisfy demand from inventory
                        inventory[item] -= demand_at_t
                        if (item, t) in remaining_demand:
                            del remaining_demand[item, t]
                    else:
                        # NoBacklog violation
                        nobacklog_violations += demand_at_t - inventory[item]
                        inventory[item] = 0

                # Stocking cost for items in inventory
                stock_cost_per_unit = (
                    self.problem.stock_cost_per_type_per_time_per_unit[item]
                )
                stocking_cost += stock_cost_per_unit * inventory[item]

        # Penalty for NoBacklog violations (high weight to enforce feasibility)
        penalty = 1000.0 * nobacklog_violations

        total_cost = stocking_cost + setup_cost + penalty

        return total_cost

    def _smart_insert_stock_aware(
        self, solution: List[int]
    ) -> Optional[Tuple[int, int]]:
        """Find smart INSERT move to reduce stock costs.

        Returns:
            Tuple (from_pos, to_pos) or None if no good move found
        """
        horizon = len(solution)

        # Find production positions
        production_positions = [
            i
            for i in range(horizon)
            if solution[i] >= 0 and solution[i] < self.problem.nb_items_type
        ]

        if len(production_positions) < 2:
            return None

        # Calculate which items have high stock cost potential
        item_early_production = {}
        for pos in production_positions:
            item = solution[pos]
            if item not in item_early_production:
                item_early_production[item] = []
            item_early_production[item].append(pos)

        # Find demands for each item
        item_demands = {}
        for item in range(self.problem.nb_items_type):
            demands = self.problem.demands[item]
            demand_times = [t for t, d in enumerate(demands) if d > 0]
            if demand_times:
                item_demands[item] = demand_times

        # Score each production: stock_cost × (deadline - production_time)
        scored_productions = []
        for item, positions in item_early_production.items():
            if item in item_demands:
                stock_cost = self.problem.stock_cost_per_type_per_time_per_unit[item]
                for pos in positions:
                    # Find closest demand after this production
                    future_demands = [d for d in item_demands[item] if d >= pos]
                    if future_demands:
                        deadline = future_demands[0]
                        stock_time = deadline - pos
                        score = stock_cost * stock_time
                        scored_productions.append((score, pos, deadline))

        if not scored_productions:
            return None

        # Pick from top 20%
        scored_productions.sort(reverse=True)
        top_n = max(1, len(scored_productions) // 5)
        score, from_pos, deadline = random.choice(scored_productions[:top_n])

        # Move closer to deadline
        possible_targets = list(range(from_pos + 1, min(deadline + 1, horizon)))
        if possible_targets:
            to_pos = random.choice(possible_targets)
            return (from_pos, to_pos)

        return None

    def _smart_swap_changeover_aware(
        self, solution: List[int]
    ) -> Optional[Tuple[int, int]]:
        """Find smart SWAP move to reduce changeover costs.

        Returns:
            Tuple (pos1, pos2) or None if no good move found
        """
        horizon = len(solution)

        # Find production positions
        production_positions = [
            i
            for i in range(horizon)
            if solution[i] >= 0 and solution[i] < self.problem.nb_items_type
        ]

        if len(production_positions) < 2:
            return None

        # Find expensive changeovers
        changeovers = []
        for idx in range(len(production_positions) - 1):
            pos1 = production_positions[idx]
            pos2 = production_positions[idx + 1]
            item1 = solution[pos1]
            item2 = solution[pos2]

            if item1 != item2:
                cost = self.problem.changeover_costs[item1][item2]
                changeovers.append((cost, pos1, pos2, item1, item2))

        if not changeovers:
            return None

        # Pick expensive changeover (top 20%)
        changeovers.sort(reverse=True)
        top_n = max(1, len(changeovers) // 5)
        cost, pos1, pos2, item1, item2 = random.choice(changeovers[:top_n])

        # Try to batch: find another production of item1 or item2
        item_to_batch = random.choice([item1, item2])
        other_positions = [
            p
            for p in production_positions
            if p not in [pos1, pos2] and solution[p] == item_to_batch
        ]

        if other_positions:
            target_pos = random.choice(other_positions)
            swap_pos = pos1 if item_to_batch == item2 else pos2
            return (swap_pos, target_pos)

        return None

    def _smart_insert_deadline_aware(
        self, solution: List[int]
    ) -> Optional[Tuple[int, int]]:
        """Find smart INSERT move to aggressively reduce deadline gaps.

        Returns:
            Tuple (from_pos, to_pos) or None if no good move found
        """
        horizon = len(solution)

        # Find production positions
        production_positions = [
            i
            for i in range(horizon)
            if solution[i] >= 0 and solution[i] < self.problem.nb_items_type
        ]

        if len(production_positions) < 1:
            return None

        # Calculate deadline gaps
        deadline_gaps = []
        for pos in production_positions:
            item = solution[pos]
            demands = self.problem.demands[item]

            # Find next demand after this production
            future_demands = [t for t in range(pos, horizon) if demands[t] > 0]
            if future_demands:
                deadline = future_demands[0]
                gap = deadline - pos
                if gap > 1:
                    deadline_gaps.append((gap, pos, deadline, item))

        if not deadline_gaps:
            return None

        # Pick from top 30%
        deadline_gaps.sort(reverse=True)
        top_n = max(1, len(deadline_gaps) * 3 // 10)
        gap, from_pos, deadline, item = random.choice(deadline_gaps[:top_n])

        # Move closer to deadline (randomly between halfway and just before deadline)
        min_target = from_pos + gap // 2
        max_target = min(deadline, horizon - 1)

        if min_target < max_target:
            to_pos = random.randint(min_target, max_target)
            return (from_pos, to_pos)

        return None

    def _get_neighbor(self, solution: List[int], move_type: MoveType) -> List[int]:
        """Generate neighbor solution.

        Args:
            solution: Current solution
            move_type: Type of move to apply

        Returns:
            New solution
        """
        new_solution = solution.copy()
        horizon = len(solution)

        # Try smart mutation if enabled
        smart_positions = None
        if self.use_smart_mutations and random.random() < self.smart_mutation_prob:
            # Try smart move based on move type
            if move_type == MoveType.INSERT:
                # Try stock-aware or deadline-aware INSERT
                if random.random() < 0.5:
                    smart_positions = self._smart_insert_stock_aware(solution)
                else:
                    smart_positions = self._smart_insert_deadline_aware(solution)
            else:  # SWAP
                # Try changeover-aware SWAP
                smart_positions = self._smart_swap_changeover_aware(solution)

        # Apply smart move if found, otherwise fall back to random
        if smart_positions is not None:
            i, j = smart_positions

            if move_type == MoveType.INSERT:
                # Move element from i to j
                element = new_solution[i]
                new_solution.pop(i)
                new_solution.insert(j, element)
            else:  # SWAP
                # Swap elements
                new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        else:
            # Random move (baseline)
            if move_type == MoveType.INSERT:
                # Select random positions
                i = random.randint(0, horizon - 1)
                j = random.randint(0, horizon - 1)

                if i != j:
                    # Move element from i to j
                    element = new_solution[i]
                    new_solution.pop(i)
                    new_solution.insert(j, element)

            elif move_type == MoveType.SWAP:
                # Select two random positions
                i = random.randint(0, horizon - 1)
                j = random.randint(0, horizon - 1)

                if i != j:
                    # Swap elements
                    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]

        return new_solution

    def _solution_to_lotsizing(self, solution: List[int]) -> LotSizingSolution:
        """Convert internal solution to LotSizingSolution.

        Args:
            solution: Production sequence

        Returns:
            LotSizingSolution object
        """
        productions = []

        # Track what has been produced
        for t, item in enumerate(solution):
            if item >= 0:
                productions.append(ProductionItem(item_type=item, quantity=1, time=t))

        # Pass deliveries=None to trigger recompute_deliveries()
        # which properly handles inventory flow and prevents negative stock
        return LotSizingSolution(
            problem=self.problem,
            productions=productions,
            deliveries=None,
        )

    def solve(
        self,
        callbacks: Optional[List[Callback]] = None,
        **kwargs: Any,
    ) -> ResultStorage:
        """Solve using Simulated Annealing.

        Args:
            callbacks: Optional callbacks for monitoring
            **kwargs: Additional arguments

        Returns:
            ResultStorage with solutions found
        """
        # Initialize
        callback = CallbackList(callbacks=callbacks)
        callback.on_solve_start(solver=self)

        result_storage = self.create_result_storage()

        # Generate initial solution
        logger.info("Generating initial solution...")
        current_solution = self._generate_initial_solution()
        current_cost = self._evaluate_solution(current_solution)

        best_solution = current_solution.copy()
        best_cost = current_cost

        logger.info(f"Initial cost: {current_cost}")
        if self.use_smart_mutations:
            logger.info(
                f"Smart mutations ENABLED (probability: {self.smart_mutation_prob:.2f})"
            )

        # SA parameters
        T = self.T0
        iteration = 0
        sampled = 0
        accepted = 0

        # Main loop
        while iteration < self.max_iterations:
            # Choose move type
            if random.random() < self.beta:
                move_type = MoveType.INSERT
            else:
                move_type = MoveType.SWAP

            # Generate neighbor
            neighbor_solution = self._get_neighbor(current_solution, move_type)
            neighbor_cost = self._evaluate_solution(neighbor_solution)

            # Calculate cost difference
            delta = neighbor_cost - current_cost

            # Accept or reject
            accept = False
            if delta <= 0:
                # Always accept improvement
                accept = True
            else:
                # Accept with probability e^(-delta/T)
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    accept = True

            if accept:
                current_solution = neighbor_solution
                current_cost = neighbor_cost
                accepted += 1

                # Update best
                if current_cost < best_cost:
                    best_solution = current_solution.copy()
                    best_cost = current_cost

                    # Store solution
                    sol = self._solution_to_lotsizing(best_solution)
                    fitness = self.aggreg_from_sol(sol)
                    result_storage.append((sol, fitness))

                    logger.info(
                        f"Iteration {iteration}: New best cost = {best_cost:.2f}"
                    )

                    # Callback
                    stopping = callback.on_step_end(
                        step=iteration, res=result_storage, solver=self
                    )
                    if stopping:
                        logger.info("Stopping criterion met")
                        break

            sampled += 1

            # Cool down temperature
            if sampled >= self.n_s or accepted >= self.n_a:
                T *= self.alpha
                logger.debug(
                    f"Temperature decreased to {T:.4f} "
                    f"(sampled={sampled}, accepted={accepted})"
                )
                sampled = 0
                accepted = 0

            iteration += 1

            # Periodic logging
            if iteration % 10000 == 0:
                logger.info(
                    f"Iteration {iteration}/{self.max_iterations}, "
                    f"Current cost: {current_cost:.2f}, "
                    f"Best cost: {best_cost:.2f}, T={T:.4f}"
                )

        # Final solution
        if len(result_storage) == 0:
            sol = self._solution_to_lotsizing(best_solution)
            fitness = self.aggreg_from_sol(sol)
            result_storage.append((sol, fitness))

        logger.info(f"SA finished. Best cost: {best_cost:.2f}")
        callback.on_solve_end(res=result_storage, solver=self)

        return result_storage
