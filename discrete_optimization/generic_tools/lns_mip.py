#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from collections.abc import Iterable
from typing import Any, Optional

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Problem,
)
from discrete_optimization.generic_tools.lns_tools import (
    BaseLns,
    ConstraintHandler,
    InitialSolution,
    PostProcessSolution,
)
from discrete_optimization.generic_tools.lp_tools import (
    GurobiMilpSolver,
    MilpSolver,
    OrtoolsMathOptMilpSolver,
    ParametersMilp,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)


class GurobiConstraintHandler(ConstraintHandler):
    def remove_constraints_from_previous_iteration(
        self,
        solver: GurobiMilpSolver,
        previous_constraints: Iterable[Any],
        **kwargs: Any,
    ) -> None:
        solver.remove_constraints(previous_constraints)


class OrtoolsMathOptConstraintHandler(ConstraintHandler):
    def remove_constraints_from_previous_iteration(
        self,
        solver: OrtoolsMathOptMilpSolver,
        previous_constraints: Iterable[Any],
        **kwargs: Any,
    ) -> None:
        solver.remove_constraints(previous_constraints)


class LnsMilp(BaseLns):
    """Large Neighborhood Search solver using a milp solver at each iteration."""

    subsolver: MilpSolver
    """Sub-solver used by this lns solver at each iteration."""

    def __init__(
        self,
        problem: Problem,
        subsolver: Optional[MilpSolver] = None,
        initial_solution_provider: Optional[InitialSolution] = None,
        constraint_handler: Optional[ConstraintHandler] = None,
        post_process_solution: Optional[PostProcessSolution] = None,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(
            problem=problem,
            subsolver=subsolver,
            initial_solution_provider=initial_solution_provider,
            constraint_handler=constraint_handler,
            post_process_solution=post_process_solution,
            params_objective_function=params_objective_function,
            **kwargs,
        )

    def init_model(self, **kwargs: Any) -> None:
        if self.subsolver.model is None:
            self.subsolver.init_model(**kwargs)
            if self.subsolver.model is None:  # for mypy
                raise RuntimeError(
                    "subsolver model must not be None after calling init_model()!"
                )

    def solve(
        self,
        nb_iteration_lns: int,
        parameters_milp: Optional[ParametersMilp] = None,
        time_limit_subsolver: Optional[float] = 30.0,
        time_limit_subsolver_iter0: Optional[float] = None,
        nb_iteration_no_improvement: Optional[int] = None,
        skip_initial_solution_provider: bool = False,
        stop_first_iteration_if_optimal: bool = True,
        callbacks: Optional[list[Callback]] = None,
        **kwargs: Any,
    ) -> ResultStorage:
        """Solve the problem with an LNS loop

        Args:
            nb_iteration_lns: number of lns iteration
            parameters_milp: parameters needed by the milp solver
            time_limit_subsolver: time limit (in seconds) for a subsolver `solve()` call
                If None, no time limit is applied.
            time_limit_subsolver_iter0: time limit (in seconds) for the first subsolver `solve()` call,
                in the case we are skipping the initial solution provider (`skip_initial_solution_provider is True`)
                If None, we use the regular `time_limit` parameter even for this first solve.
            nb_iteration_no_improvement: maximal number of consecutive iteration without improvement allowed
                before stopping the solve process.
            skip_initial_solution_provider: if True, we do not use `self.initial_solution_provider`
                but instead launch a first `self.subsolver.solve()`
            stop_first_iteration_if_optimal: if True, if `skip_initial_solution_provider, and if subsolver tells
                its result is optimal after the first `self.subsolver.solve()` (so before any constraint tempering),
                we stop the solve process.
            callbacks: list of callbacks used to hook into the various stage of the solve
            **kwargs: passed to the subsolver

        Returns:

        """
        if parameters_milp is None:
            parameters_milp = ParametersMilp.default()
        return super().solve(
            parameters_milp=parameters_milp,
            nb_iteration_lns=nb_iteration_lns,
            time_limit_subsolver=time_limit_subsolver,
            time_limit_subsolver_iter0=time_limit_subsolver_iter0,
            nb_iteration_no_improvement=nb_iteration_no_improvement,
            stop_first_iteration_if_optimal=stop_first_iteration_if_optimal,
            skip_initial_solution_provider=skip_initial_solution_provider,
            callbacks=callbacks,
            **kwargs,
        )
