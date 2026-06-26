#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Sequential horizon solver for lot sizing problems.

This solver incrementally increases the planning horizon, solving sub-problems
and using previous solutions as warmstart for the next iteration.

Strategy:
    1. Start with a small horizon (e.g., 20% of total)
    2. Solve the subproblem
    3. Increase horizon incrementally
    4. Extend previous solution with greedy for new periods (warmstart)
    5. Solve expanded problem
    6. Repeat until full horizon is reached

This approach is particularly effective for:
    - Large horizon problems where full solve is too slow
    - Problems where early periods constrain later decisions
    - Any solver that benefits from good initial solutions
"""

import logging
from typing import Any, Optional, Type

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.lotsizing.problem import (
    LotSizingProblem,
    LotSizingSolution,
)
from discrete_optimization.lotsizing.solvers.greedy import (
    GreedyLotSizingSolver,
    GreedyStrategy,
)

logger = logging.getLogger(__name__)


def create_subproblem(problem: LotSizingProblem, horizon: int) -> LotSizingProblem:
    """Create a subproblem with reduced horizon.

    Args:
        problem: Original problem
        horizon: New horizon (must be <= problem.horizon)

    Returns:
        New problem instance with horizon periods
    """
    if horizon > problem.horizon:
        raise ValueError(
            f"New horizon {horizon} exceeds problem horizon {problem.horizon}"
        )

    # Truncate demands to new horizon
    demands_truncated = [
        problem.demands[item][:horizon] for item in range(problem.nb_items_type)
    ]

    return LotSizingProblem(
        nb_items_type=problem.nb_items_type,
        capacity_machine=problem.capacity_machine,
        changeover_costs=problem.changeover_costs,
        demands=demands_truncated,
        stock_capacity=problem.stock_capacity,
        stock_cost_per_type_per_time_per_unit=problem.stock_cost_per_type_per_time_per_unit,
        delay_cost_per_type_per_time_per_unit=problem.delay_cost_per_type_per_time_per_unit,
        allow_delays=problem.allow_delays,
    )


def extend_solution_with_greedy(
    solution: LotSizingSolution,
    target_problem: LotSizingProblem,
    greedy_strategy: GreedyStrategy = GreedyStrategy.BALANCED,
) -> LotSizingSolution:
    """Extend a solution to a larger horizon using greedy for new periods.

    Args:
        solution: Solution for a problem with smaller horizon
        target_problem: Problem with larger horizon
        greedy_strategy: Strategy for greedy construction of new periods

    Returns:
        Extended solution for target_problem
    """
    original_horizon = len(solution.list_item_per_time)
    target_horizon = target_problem.horizon

    if original_horizon >= target_horizon:
        # No extension needed, just convert to target problem
        return LotSizingSolution(
            problem=target_problem,
            list_item_per_time=solution.list_item_per_time[:target_horizon],
        )

    # Create a greedy solution for the full target problem
    greedy_solver = GreedyLotSizingSolver(target_problem)
    greedy_result = greedy_solver.solve(strategy=greedy_strategy)
    greedy_solution = greedy_result[0][0]

    # Combine: keep original solution for existing periods, use greedy for new periods
    extended_list = (
        solution.list_item_per_time[:original_horizon]
        + greedy_solution.list_item_per_time[original_horizon:]
    )

    return LotSizingSolution(problem=target_problem, list_item_per_time=extended_list)


class SequentialHorizonSolver(SolverDO):
    """Sequential horizon solver for lot sizing.

    Solves the problem by incrementally increasing the horizon, using previous
    solutions as warmstart for the next iteration.

    Args:
        problem: The lot sizing problem to solve
        subsolver_cls: Solver class to use for each subproblem
        subsolver_kwargs: Keyword arguments for the subsolver
        initial_horizon: Starting horizon (default: 20% of total)
        horizon_increment: How much to increase horizon each step (default: 20% of total)
        greedy_strategy: Strategy for extending solutions with greedy

    Example:
        ```python
        from discrete_optimization.lotsizing.solvers.cpsat import CpSatLotSizingSolver
        from discrete_optimization.lotsizing.solvers.sequential_horizon import SequentialHorizonSolver

        # Solve incrementally using CP-SAT on each chunk
        solver = SequentialHorizonSolver(
            problem,
            subsolver_cls=CpSatLotSizingSolver,
            subsolver_kwargs={"time_limit": 30},
            initial_horizon=50,
            horizon_increment=50,
        )
        result = solver.solve()
        ```
    """

    problem: LotSizingProblem

    def __init__(
        self,
        problem: LotSizingProblem,
        subsolver_cls: Type[SolverDO],
        subsolver_kwargs: Optional[dict] = None,
        initial_horizon: Optional[int] = None,
        horizon_increment: Optional[int] = None,
        greedy_strategy: GreedyStrategy = GreedyStrategy.BALANCED,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)

        self.subsolver_cls = subsolver_cls
        self.subsolver_kwargs = subsolver_kwargs or {}

        # Default: start with 20% of horizon
        self.initial_horizon = initial_horizon or max(10, problem.horizon // 5)

        # Default: increment by 20% of horizon
        self.horizon_increment = horizon_increment or max(10, problem.horizon // 5)

        self.greedy_strategy = greedy_strategy

        # Ensure initial horizon is valid
        self.initial_horizon = min(self.initial_horizon, problem.horizon)

    def solve(
        self,
        callbacks: Optional[list[Callback]] = None,
        **kwargs: Any,
    ) -> ResultStorage:
        """Solve using sequential horizon approach.

        Returns:
            ResultStorage with best solution found
        """
        callback = CallbackList(callbacks=callbacks)
        callback.on_solve_start(self)

        result_storage = self.create_result_storage()

        print("=" * 70)
        print("SEQUENTIAL HORIZON SOLVER - Starting")
        print("=" * 70)
        print(
            f"Problem: {self.problem.nb_items_type} items, {self.problem.horizon} periods"
        )
        print(f"Subsolver: {self.subsolver_cls.__name__}")
        print(f"Initial horizon: {self.initial_horizon}")
        print(f"Horizon increment: {self.horizon_increment}")
        print(f"Greedy strategy for extension: {self.greedy_strategy.value}")
        print()

        current_horizon = self.initial_horizon
        best_solution = None
        iteration = 0

        while current_horizon <= self.problem.horizon:
            iteration += 1
            print(f"{'=' * 70}")
            print(
                f"Iteration {iteration}: Solving horizon 0..{current_horizon} (out of {self.problem.horizon})"
            )
            print(f"{'=' * 70}")

            # Create subproblem
            subproblem = create_subproblem(self.problem, current_horizon)

            # Create subsolver
            subsolver = self.subsolver_cls(
                subproblem,
                params_objective_function=self.params_objective_function,
                **self.subsolver_kwargs,
            )

            # Prepare initial solution if we have a previous solution
            initial_solution = None
            if best_solution is not None:
                # Extend previous solution to current horizon
                initial_solution = extend_solution_with_greedy(
                    best_solution, subproblem, self.greedy_strategy
                )

                # Evaluate warmstart quality
                warmstart_fit = subproblem.evaluate(initial_solution)
                warmstart_cost = sum(warmstart_fit.values())
                print(f"Warmstart solution cost: {warmstart_cost:.2f}")
                print(
                    f"  (extended from horizon {len(best_solution.list_item_per_time)} using {self.greedy_strategy.value})"
                )

            # Solve subproblem
            # Pass initial solution if the solver supports it via kwargs
            solve_kwargs = kwargs.copy()
            if initial_solution is not None and hasattr(
                subsolver, "_supports_initial_solution"
            ):
                solve_kwargs["initial_solution"] = initial_solution
            solve_kwargs.update(self.subsolver_kwargs)

            print(f"Solving subproblem...")
            subresult = subsolver.solve(**solve_kwargs)

            # Get best solution from subproblem
            subsol, subfit = subresult[0]
            print(f"Subproblem solved: cost = {subfit:.2f}")

            # Convert solution to full problem domain
            if current_horizon < self.problem.horizon:
                # Extend to full horizon for storage and next iteration
                best_solution = extend_solution_with_greedy(
                    subsol, self.problem, self.greedy_strategy
                )
                fit = self.aggreg_from_sol(best_solution)
                print(f"Extended to full horizon: cost = {fit:.2f}")
            else:
                # Final iteration - this is the full problem
                best_solution = LotSizingSolution(
                    problem=self.problem, list_item_per_time=subsol.list_item_per_time
                )
                fit = subfit

            # Store result
            result_storage.append((best_solution, fit))

            print()

            # Prepare for next iteration
            if current_horizon >= self.problem.horizon:
                break

            # Increase horizon
            prev_horizon = current_horizon
            current_horizon = min(
                current_horizon + self.horizon_increment, self.problem.horizon
            )

            if current_horizon == prev_horizon:
                # No progress, stop
                break

        print("=" * 70)
        print("SEQUENTIAL HORIZON SOLVER - Completed")
        print("=" * 70)
        print(f"Total iterations: {iteration}")
        print(f"Final solution cost: {fit:.2f}")
        print()

        callback.on_solve_end(res=result_storage, solver=self)

        return result_storage


def create_sequential_solver(
    problem: LotSizingProblem,
    subsolver_cls: Type[SolverDO],
    subsolver_kwargs: Optional[dict] = None,
    num_chunks: int = 5,
    greedy_strategy: GreedyStrategy = GreedyStrategy.BALANCED,
    **kwargs,
) -> SequentialHorizonSolver:
    """Convenience function to create a sequential horizon solver.

    Args:
        problem: The lot sizing problem
        subsolver_cls: Solver class to use on subproblems
        subsolver_kwargs: Kwargs for subsolver
        num_chunks: Number of chunks to divide horizon into (default: 5)
        greedy_strategy: Strategy for extending solutions
        **kwargs: Additional kwargs for SequentialHorizonSolver

    Returns:
        Configured SequentialHorizonSolver

    Example:
        ```python
        from discrete_optimization.lotsizing.solvers.cpsat import CpSatLotSizingSolver

        solver = create_sequential_solver(
            problem,
            subsolver_cls=CpSatLotSizingSolver,
            subsolver_kwargs={"time_limit": 60},
            num_chunks=5,
        )
        result = solver.solve()
        ```
    """
    chunk_size = max(10, problem.horizon // num_chunks)

    return SequentialHorizonSolver(
        problem,
        subsolver_cls=subsolver_cls,
        subsolver_kwargs=subsolver_kwargs,
        initial_horizon=chunk_size,
        horizon_increment=chunk_size,
        greedy_strategy=greedy_strategy,
        **kwargs,
    )
