#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Solver wrapper that applies problem transformation before solving."""

from __future__ import annotations

from typing import Any, Optional

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Problem,
    Solution,
)
from discrete_optimization.generic_tools.do_solver import SolverDO, WarmstartMixin
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.generic_tools.transformation.problem_transformation import (
    ProblemTransformation,
)


class TransformationSolver(SolverDO, WarmstartMixin):
    """Solver that applies problem transformation before solving.

    This wrapper:
    1. Transforms P1 → P2 using transformation (or uses pre-computed target_problem)
    2. Instantiates and solves P2 using solver_brick
    3. Back-transforms solutions S2 → S1
    4. Stores solutions as S1 in ResultStorage

    Example (simple):
        >>> transformation = RcpspToMultiskillTransformation()
        >>> solver = TransformationSolver(
        ...     transformation=transformation,
        ...     source_problem=rcpsp_problem,
        ...     solver_brick=SubBrick(
        ...         cls=CPSatMultiskillSolver,
        ...         kwargs={"time_limit": 60}
        ...     )
        ... )
        >>> result = solver.solve()  # Returns RcpspSolution!

    Example (with pre-computed target problem):
        >>> transformation = RcpspToMultiskillTransformation()
        >>> target_problem = transformation.transform_problem(rcpsp_problem)
        >>> solver = TransformationSolver(
        ...     transformation=transformation,
        ...     source_problem=rcpsp_problem,
        ...     target_problem=target_problem,  # Already transformed!
        ...     solver_brick=SubBrick(...)
        ... )

    Example (for use in SequentialMetasolver):
        >>> # TransformationSolver also accepts 'problem' arg for SolverDO compatibility
        >>> solver = TransformationSolver(
        ...     problem=rcpsp_problem,  # Used as source_problem
        ...     transformation=RcpspToMultiskillTransformation(),
        ...     solver_brick=SubBrick(...)
        ... )

    Note: Callbacks currently receive transformed (target) solutions.
    Future enhancement will back-transform solutions for callbacks.

    """

    transformation: ProblemTransformation
    source_problem: Problem
    target_problem: Problem
    wrapped_solver: SolverDO
    _solution_cache: dict[int, Solution]  # Cache for back-transformed solutions

    def __init__(
        self,
        transformation: ProblemTransformation,
        solver_brick: SubBrick,
        source_problem: Optional[Problem] = None,
        target_problem: Optional[Problem] = None,
        problem: Optional[Problem] = None,  # Alias for source_problem (SolverDO compat)
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        """Initialize transformation solver.

        Args:
            transformation: Transformation to apply (stateless)
            solver_brick: SubBrick defining solver class + kwargs for target problem
            source_problem: Source problem (P1) - provide this OR problem
            target_problem: Target problem (P2) - optional, computed if not provided
            problem: Alias for source_problem (for SolverDO/SequentialMetasolver compatibility)
            params_objective_function: Objective function params for SOURCE problem (P1)
            **kwargs: Additional arguments

        Raises:
            ValueError: If source_problem not provided

        """
        # Determine source problem (accept either source_problem or problem)
        if source_problem is not None and problem is not None:
            if source_problem != problem:
                raise ValueError(
                    "source_problem and problem are different. Provide only one."
                )
            self.source_problem = source_problem
        elif source_problem is not None:
            self.source_problem = source_problem
        elif problem is not None:
            self.source_problem = problem
        else:
            raise ValueError("Must provide either source_problem or problem")

        # Initialize SolverDO with SOURCE problem
        super().__init__(
            problem=self.source_problem,
            params_objective_function=params_objective_function,
            **kwargs,
        )

        self.transformation = transformation

        # Get or create target problem
        if target_problem is not None:
            self.target_problem = target_problem
        else:
            self.target_problem = self.transformation.transform_problem(
                self.source_problem
            )

        # Instantiate solver for target problem
        self.solver_brick = solver_brick
        self.wrapped_solver = solver_brick.cls(
            problem=self.target_problem, **solver_brick.kwargs
        )

        # Cache for solution transformations (S2 id -> S1)
        self._solution_cache = {}

    def init_model(self, **kwargs: Any) -> None:
        """Initialize the wrapped solver's model.

        Args:
            **kwargs: Arguments passed to wrapped solver's init_model

        """
        self.wrapped_solver.init_model(**kwargs)

    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        """Solve by transforming, solving, and back-transforming.

        Args:
            callbacks: Callbacks (currently receive target problem solutions)
            **kwargs: Arguments passed to wrapped solver

        Returns:
            ResultStorage containing SOURCE problem solutions (S1)

        """
        # TODO: Wrap callbacks to back-transform solutions
        # For now, callbacks receive target solutions

        # Solve in transformed space
        merged_kwargs = kwargs.copy()
        merged_kwargs.update(self.solver_brick.kwargs)
        target_result = self.wrapped_solver.solve(callbacks=callbacks, **merged_kwargs)

        # Back-transform all solutions
        source_result = self.create_result_storage()
        for solution_target, _ in target_result:
            solution_source = self._get_or_transform_solution(solution_target)
            # Re-evaluate in source problem to ensure consistency
            source_result.append(
                (solution_source, self.aggreg_from_sol(solution_source))
            )

        # Update status
        self.status_solver = self.wrapped_solver.status_solver

        # Clear cache
        self._solution_cache.clear()

        return source_result

    def _get_or_transform_solution(self, solution_target: Solution) -> Solution:
        """Get transformed solution from cache or transform and cache it.

        This avoids re-transforming the same solution multiple times.

        Args:
            solution_target: Solution in target problem space

        Returns:
            Solution in source problem space

        """
        sol_id = id(solution_target)
        if sol_id not in self._solution_cache:
            self._solution_cache[sol_id] = self.transformation.back_transform_solution(
                solution_target, self.source_problem
            )
        return self._solution_cache[sol_id]

    def set_warm_start(self, solution: Solution) -> None:
        """Set warmstart solution from SOURCE problem.

        Args:
            solution: Solution in source problem space (S1)

        Raises:
            ValueError: If transformation doesn't support forward transformation
                       or wrapped solver doesn't support warmstart

        """
        if not isinstance(self.wrapped_solver, WarmstartMixin):
            raise ValueError("Wrapped solver does not support warmstart")

        # Forward-transform solution
        target_solution = self.transformation.forward_transform_solution(
            solution, self.target_problem
        )

        if target_solution is None:
            raise ValueError(
                f"Transformation {type(self.transformation).__name__} "
                "does not support forward transformation (warmstart not available)"
            )

        self.wrapped_solver.set_warm_start(target_solution)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TransformationSolver({self.transformation}, "
            f"solver={type(self.wrapped_solver).__name__})"
        )
