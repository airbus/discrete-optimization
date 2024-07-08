#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Iterable, List, Optional

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Problem,
)
from discrete_optimization.generic_tools.lns_tools import (
    BaseLNS,
    ConstraintHandler,
    InitialSolution,
    PostProcessSolution,
)
from discrete_optimization.generic_tools.lp_tools import (
    GurobiMilpSolver,
    MilpSolver,
    MilpSolverName,
    ParametersMilp,
    PymipMilpSolver,
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
        solver.model.remove(list(previous_constraints))
        solver.model.update()


class PymipConstraintHandler(ConstraintHandler):
    def remove_constraints_from_previous_iteration(
        self,
        solver: PymipMilpSolver,
        previous_constraints: Iterable[Any],
        **kwargs: Any,
    ) -> None:
        solver.model.remove(list(previous_constraints))
        if solver.milp_solver_name == MilpSolverName.GRB:
            solver.model.solver.update()


class LNS_MILP(BaseLNS):
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
        nb_iteration_no_improvement: Optional[int] = None,
        skip_initial_solution_provider: bool = False,
        stop_first_iteration_if_optimal: bool = True,
        callbacks: Optional[List[Callback]] = None,
        **kwargs: Any,
    ) -> ResultStorage:
        if parameters_milp is None:
            parameters_milp = ParametersMilp.default()
        return super().solve(
            parameters_milp=parameters_milp,
            nb_iteration_lns=nb_iteration_lns,
            nb_iteration_no_improvement=nb_iteration_no_improvement,
            stop_first_iteration_if_optimal=stop_first_iteration_if_optimal,
            skip_initial_solution_provider=skip_initial_solution_provider,
            callbacks=callbacks,
            **kwargs,
        )
