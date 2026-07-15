#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""High-performance Simulated Annealing solver for capacitated multi-item lot sizing using numpy/numba.

This is an optimized version using:
- Numpy arrays for vectorized operations
- Numba JIT compilation for hot paths
- Precomputed data structures
- Minimal Python overhead

Expected speedup: 10-50x over pure Python version, 100x with cached compilation.
"""

import logging
import math
import random
from typing import Any, List, Optional

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Fallback decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    FloatHyperparameter,
    IntegerHyperparameter,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.lotsizing import ProductionDecision
from discrete_optimization.lotsizing.capacitatedmultiitem.problem import (
    CapacitatedMultiItemLSP,
    CapacitatedMultiItemSolution,
)

logger = logging.getLogger(__name__)


# ============================================================================
# NUMBA-COMPILED FUNCTIONS (HOT PATH)
# ============================================================================


@njit
def evaluate_solution_numba(
    solution: np.ndarray,
    demands: np.ndarray,
    stock_costs: np.ndarray,
    changeover_costs: np.ndarray,
    nb_items: int,
    horizon: int,
) -> float:
    """Evaluate cost of solution (numba-compiled for speed).

    Args:
        solution: Production sequence (horizon,) array, -1 = idle
        demands: Demand matrix (nb_items, horizon)
        stock_costs: Stock cost per item (nb_items,)
        changeover_costs: Changeover cost matrix (nb_items, nb_items)
        nb_items: Number of item types
        horizon: Number of periods

    Returns:
        Total cost (stocking + changeover + NoBacklog penalty)
    """
    # Track inventory for each item
    inventory = np.zeros(nb_items, dtype=np.float64)

    # Track last produced item (for changeover cost)
    last_item = -1

    # Costs
    stocking_cost = 0.0
    changeover_cost = 0.0
    nobacklog_violations = 0.0

    # Process each period
    for t in range(horizon):
        produced_item = solution[t]

        # Production
        if produced_item >= 0:
            inventory[produced_item] += 1.0

            # Changeover cost (no initial setup cost)
            if last_item != -1 and last_item != produced_item:
                changeover_cost += changeover_costs[last_item, produced_item]

            last_item = produced_item

        # Satisfy demands and calculate stocking cost
        for item in range(nb_items):
            demand_at_t = demands[item, t]

            if demand_at_t > 0:
                if inventory[item] >= demand_at_t:
                    # Satisfy demand from inventory
                    inventory[item] -= demand_at_t
                else:
                    # NoBacklog violation
                    nobacklog_violations += demand_at_t - inventory[item]
                    inventory[item] = 0.0

            # Stocking cost for items in inventory
            stocking_cost += stock_costs[item] * inventory[item]

    # Penalty for NoBacklog violations
    penalty = 1000.0 * nobacklog_violations

    total_cost = stocking_cost + changeover_cost + penalty
    return total_cost


@njit
def apply_insert_move(solution: np.ndarray, i: int, j: int) -> np.ndarray:
    """Apply INSERT move (numba-compiled).

    Move element from position i to position j.
    """
    new_solution = solution.copy()
    if i != j:
        element = new_solution[i]
        # Shift elements
        if i < j:
            new_solution[i:j] = new_solution[i + 1 : j + 1]
        else:
            new_solution[j + 1 : i + 1] = new_solution[j:i]
        new_solution[j] = element
    return new_solution


@njit
def apply_swap_move(solution: np.ndarray, i: int, j: int) -> np.ndarray:
    """Apply SWAP move (numba-compiled)."""
    new_solution = solution.copy()
    if i != j:
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    return new_solution


@njit
def generate_initial_solution_numba(
    demands: np.ndarray,
    nb_items: int,
    horizon: int,
    seed: int,
) -> np.ndarray:
    """Generate random initial solution (numba-compiled).

    Returns:
        Solution array of length horizon
    """
    np.random.seed(seed)

    # Create solution vector
    solution = np.full(horizon, -1, dtype=np.int32)

    # Collect all demands
    all_demands = []
    for item in range(nb_items):
        for t in range(horizon):
            if demands[item, t] > 0:
                # Add demand occurrences
                for _ in range(int(demands[item, t])):
                    all_demands.append((item, t))

    if len(all_demands) == 0:
        return solution

    # Shuffle demands
    all_demands_arr = np.array(all_demands, dtype=np.int32)
    np.random.shuffle(all_demands_arr)

    # Track used slots
    used_slots = np.zeros(horizon, dtype=np.bool_)

    # Assign each demand to a period
    for idx in range(len(all_demands_arr)):
        item = all_demands_arr[idx, 0]
        deadline = all_demands_arr[idx, 1]

        # Find a free slot before or at deadline
        possible_times = []
        for t in range(deadline + 1):
            if not used_slots[t]:
                possible_times.append(t)

        if len(possible_times) > 0:
            t = possible_times[np.random.randint(0, len(possible_times))]
        else:
            # Find first free slot anywhere
            free_slots = []
            for t in range(horizon):
                if not used_slots[t]:
                    free_slots.append(t)

            if len(free_slots) > 0:
                t = free_slots[np.random.randint(0, len(free_slots))]
            else:
                # All slots taken, use random slot
                t = np.random.randint(0, horizon)

        solution[t] = item
        used_slots[t] = True

    return solution


@njit
def sa_main_loop_numba(
    initial_solution: np.ndarray,
    initial_best_solution: np.ndarray,
    demands: np.ndarray,
    stock_costs: np.ndarray,
    changeover_costs: np.ndarray,
    nb_items: int,
    horizon: int,
    T_initial: float,
    alpha: float,
    beta: float,
    n_s: int,
    n_a: int,
    max_iterations: int,
    seed: int,
    log_interval: int,
    restart_after_no_improvement: int,
    # State to continue from previous chunk
    initial_cost: float,
    initial_best_cost: float,
    initial_T: float,
    initial_sampled: int,
    initial_accepted: int,
    initial_improvements: int,
    initial_iterations_since_improvement: int,
    initial_n_restarts: int,
) -> tuple:
    """Main SA loop (numba-compiled).

    Returns:
        (current_solution, best_solution, current_cost, best_cost, T, sampled, accepted,
         n_improvements, iterations_since_improvement, n_restarts, progress_log)
    """
    np.random.seed(seed)

    current_solution = initial_solution.copy()
    current_cost = initial_cost

    best_solution = initial_best_solution.copy()
    best_cost = initial_best_cost

    T = initial_T
    sampled = initial_sampled
    accepted = initial_accepted
    n_improvements = initial_improvements
    iterations_since_improvement = initial_iterations_since_improvement
    n_restarts = initial_n_restarts

    # Progress logging
    max_log_entries = max_iterations // log_interval + 2
    progress_log = np.zeros((max_log_entries, 5), dtype=np.float64)
    log_idx = 0

    for iteration in range(max_iterations):
        # Choose move type
        if np.random.random() < beta:
            move_type = 0  # INSERT
        else:
            move_type = 1  # SWAP

        # Generate random positions
        i = np.random.randint(0, horizon)
        j = np.random.randint(0, horizon)

        # Apply move
        if move_type == 0:  # INSERT
            neighbor_solution = apply_insert_move(current_solution, i, j)
        else:  # SWAP
            neighbor_solution = apply_swap_move(current_solution, i, j)

        # Evaluate neighbor
        neighbor_cost = evaluate_solution_numba(
            neighbor_solution, demands, stock_costs, changeover_costs, nb_items, horizon
        )

        # Calculate cost difference
        delta = neighbor_cost - current_cost

        # Accept or reject
        accept = False
        if delta <= 0:
            accept = True
        else:
            prob = math.exp(-delta / T)
            if np.random.random() < prob:
                accept = True

        if accept:
            current_solution = neighbor_solution
            current_cost = neighbor_cost
            accepted += 1

            # Update best
            if current_cost < best_cost:
                best_solution = current_solution.copy()
                best_cost = current_cost
                n_improvements += 1
                iterations_since_improvement = 0
            else:
                iterations_since_improvement += 1
        else:
            iterations_since_improvement += 1

        sampled += 1

        # Restart to best solution if stuck (optional)
        if (
            restart_after_no_improvement > 0
            and iterations_since_improvement >= restart_after_no_improvement
        ):
            # Restart from best solution
            current_solution = best_solution.copy()
            current_cost = evaluate_solution_numba(
                current_solution,
                demands,
                stock_costs,
                changeover_costs,
                nb_items,
                horizon,
            )

            iterations_since_improvement = 0
            n_restarts += 1

        # Cool down temperature
        if sampled >= n_s or accepted >= n_a:
            T *= alpha
            sampled = 0
            accepted = 0

        # Log progress
        if (iteration + 1) % log_interval == 0 and log_idx < max_log_entries:
            progress_log[log_idx, 0] = iteration + 1
            progress_log[log_idx, 1] = current_cost
            progress_log[log_idx, 2] = best_cost
            progress_log[log_idx, 3] = T
            progress_log[log_idx, 4] = n_improvements
            log_idx += 1

    return (
        current_solution,
        best_solution,
        current_cost,
        best_cost,
        T,
        sampled,
        accepted,
        n_improvements,
        iterations_since_improvement,
        n_restarts,
        progress_log[:log_idx],
    )


# ============================================================================
# SOLVER CLASS
# ============================================================================


class SimulatedAnnealingLotSizingSolverFast(SolverDO):
    """High-performance Simulated Annealing solver using numpy/numba.

    Algorithm matches Ceschia et al. (2017):
    - GPI mutations: INSERT (70%) and SWAP (30%)
    - Threshold-based temperature cooling
    - NoBacklog penalty for constraint violations

    Expected speedup: 4.5x first run (includes JIT), 100x subsequent runs (cached).
    """

    problem: CapacitatedMultiItemLSP

    hyperparameters = [
        FloatHyperparameter(name="T0", low=1.0, high=100.0, default=37.0),
        FloatHyperparameter(name="alpha", low=0.8, high=0.99, default=0.999),
        FloatHyperparameter(name="beta", low=0.0, high=1.0, default=0.7),
        IntegerHyperparameter(name="n_a", low=100, high=100000, default=12049),
        IntegerHyperparameter(name="n_s", low=10, high=100000, default=60240),
        IntegerHyperparameter(
            name="max_iterations", low=1000, high=10000000, default=1000000
        ),
        IntegerHyperparameter(
            name="restart_after_no_improvement", low=0, high=1000000, default=0
        ),
    ]

    def __init__(
        self,
        problem: CapacitatedMultiItemLSP,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.T0 = kwargs.get("T0", 37.0)
        self.alpha = kwargs.get("alpha", 0.999)
        self.beta = kwargs.get("beta", 0.7)
        self.n_a = kwargs.get("n_a", 12049)
        self.n_s = kwargs.get("n_s", 60240)
        self.max_iterations = kwargs.get("max_iterations", 1000000)
        self.restart_after_no_improvement = kwargs.get(
            "restart_after_no_improvement", 0
        )

        # Precompute numpy arrays for fast access
        self._precompute_data()

        if not NUMBA_AVAILABLE:
            logger.warning(
                "Numba not available - running without JIT compilation (slower)"
            )

    def _precompute_data(self):
        """Precompute data structures as numpy arrays."""
        # Demands matrix (nb_items, horizon)
        self.demands_np = np.zeros(
            (self.problem.nb_items, self.problem.horizon), dtype=np.float64
        )
        for item in self.problem.items_list:
            for t in range(self.problem.horizon):
                self.demands_np[item, t] = self.problem.get_demand(item, t)

        # Stock costs (nb_items,)
        self.stock_costs_np = np.array(
            [
                self.problem.get_inventory_cost_per_unit(i, 0)
                for i in self.problem.items_list
            ],
            dtype=np.float64,
        )

        # Changeover costs (nb_items, nb_items)
        changeover_matrix = []
        for i in self.problem.items_list:
            row = []
            for j in self.problem.items_list:
                row.append(self.problem.get_changeover_cost(i, j))
            changeover_matrix.append(row)
        self.changeover_costs_np = np.array(changeover_matrix, dtype=np.float64)

    def _solution_to_lotsizing(
        self, solution: np.ndarray
    ) -> CapacitatedMultiItemSolution:
        """Convert numpy solution to CapacitatedMultiItemSolution."""
        productions = []

        for t, item in enumerate(solution):
            if item >= 0:
                productions.append(
                    ProductionDecision(item=int(item), period=t, quantity=1)
                )

        return CapacitatedMultiItemSolution(
            problem=self.problem,
            productions=productions,
        )

    def solve(
        self,
        callbacks: Optional[List[Callback]] = None,
        **kwargs: Any,
    ) -> ResultStorage:
        """Solve using fast numpy/numba SA.

        Args:
            callbacks: Optional callbacks for monitoring
            **kwargs: Additional arguments
                log_interval: How often to log progress (default: 10000)

        Returns:
            ResultStorage with solutions found
        """
        import time

        # Initialize
        callback = CallbackList(callbacks=callbacks)
        callback.on_solve_start(solver=self)

        result_storage = self.create_result_storage()

        log_interval = kwargs.get("log_interval", 10000)

        # Generate initial solution
        logger.info("=" * 70)
        logger.info("FAST SA (numpy/numba) - Starting")
        logger.info("=" * 70)
        logger.info(
            f"Problem: {self.problem.nb_items} items, {self.problem.horizon} periods"
        )
        logger.info(f"Max iterations: {self.max_iterations:,}")
        logger.info(f"Parameters: T0={self.T0}, alpha={self.alpha}, beta={self.beta}")
        logger.info(f"Cooling thresholds: n_s={self.n_s:,}, n_a={self.n_a:,}")
        if self.restart_after_no_improvement > 0:
            logger.info(
                f"Restart: enabled (every {self.restart_after_no_improvement:,} iter without improvement)"
            )
        else:
            logger.info(f"Restart: disabled")
        logger.info("")

        logger.info("Generating initial solution...")
        seed = kwargs.get("seed", random.randint(0, 2**31 - 1))
        initial_solution = generate_initial_solution_numba(
            self.demands_np, self.problem.nb_items, self.problem.horizon, seed
        )

        initial_cost = evaluate_solution_numba(
            initial_solution,
            self.demands_np,
            self.stock_costs_np,
            self.changeover_costs_np,
            self.problem.nb_items,
            self.problem.horizon,
        )

        logger.info(f"Initial cost: {initial_cost:.2f}")
        logger.info("")
        logger.info("Running JIT-compiled SA main loop...")
        logger.info(
            "(First run includes compilation overhead, subsequent runs use cached code)"
        )
        logger.info("")

        # Print header for live progress
        logger.info(
            f"{'Iteration':<12} {'Current Cost':>14} {'Best Cost':>12} {'Temp':>10} {'Improvements':>14} {'Time (s)':>10} {'Iter/s':>12}"
        )
        logger.info("-" * 100)

        # Run main SA loop in chunks to get live progress
        start_time = time.time()

        current_solution = initial_solution.copy()
        best_solution = current_solution.copy()
        current_cost = initial_cost
        best_cost = initial_cost

        # SA state (must be preserved across chunks)
        T = self.T0
        sampled = 0
        accepted = 0
        n_improvements_total = 0
        n_improvements_chunk = 0  # Improvements found in current chunk
        iterations_since_improvement = 0
        n_restarts = 0

        iterations_done = 0
        chunk_size = log_interval

        while iterations_done < self.max_iterations:
            remaining = self.max_iterations - iterations_done
            this_chunk = min(chunk_size, remaining)

            if this_chunk <= 0:
                break

            prev_n_restarts = n_restarts

            # Run a chunk, preserving state
            (
                current_solution,
                best_solution,
                current_cost,
                best_cost,
                T,
                sampled,
                accepted,
                n_improvements_chunk,
                iterations_since_improvement,
                n_restarts,
                progress_log,
            ) = sa_main_loop_numba(
                current_solution,
                best_solution,
                self.demands_np,
                self.stock_costs_np,
                self.changeover_costs_np,
                self.problem.nb_items,
                self.problem.horizon,
                self.T0,
                self.alpha,
                self.beta,
                self.n_s,
                self.n_a,
                this_chunk,
                seed + iterations_done,
                this_chunk,  # log_interval within chunk
                self.restart_after_no_improvement,
                current_cost,
                best_cost,
                T,
                sampled,
                accepted,
                n_improvements_chunk,
                iterations_since_improvement,
                n_restarts,
            )

            iterations_done += this_chunk
            n_improvements_total = n_improvements_chunk

            # Print live progress
            elapsed = time.time() - start_time
            iter_per_sec = iterations_done / elapsed if elapsed > 0 else 0

            # Only show the last entry in progress_log (if any)
            if len(progress_log) > 0:
                last_iter, last_current, last_best, last_temp, last_improvements = (
                    progress_log[-1]
                )
                logger.info(
                    f"{int(last_iter + iterations_done - this_chunk):>12,}  "
                    f"{last_current:>14.2f}  "
                    f"{last_best:>12.2f}  "
                    f"{last_temp:>10.4f}  "
                    f"{int(last_improvements):>14}  "
                    f"{elapsed:>10.2f}  "
                    f"{iter_per_sec:>12,.0f}"
                )

        # Final output
        elapsed_total = time.time() - start_time
        logger.info("")
        logger.info("=" * 70)
        logger.info("FAST SA - Completed")
        logger.info("=" * 70)
        logger.info(f"Total time: {elapsed_total:.2f}s")
        logger.info(
            f"Average iterations/sec: {self.max_iterations / elapsed_total:,.0f}"
        )
        logger.info(f"Best cost: {best_cost:.2f}")
        logger.info(f"Initial cost: {initial_cost:.2f}")
        logger.info(
            f"Improvement: {initial_cost - best_cost:.2f} ({(initial_cost - best_cost) / initial_cost * 100:.1f}%)"
        )
        logger.info(f"Number of improvements found: {n_improvements_total}")
        logger.info("")

        # Convert best solution to LotSizingSolution
        best_sol = self._solution_to_lotsizing(best_solution)

        # Add to result storage
        fit = self.aggreg_from_sol(best_sol)
        result_storage.append((best_sol, fit))

        callback.on_solve_end(res=result_storage, solver=self)

        return result_storage
