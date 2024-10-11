#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import contextlib
import logging
import random
import sys
from abc import abstractmethod
from collections.abc import Iterable
from typing import Any, Optional

import numpy as np
from minizinc import Instance
from ortools.sat.python.cp_model import Constraint

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.cp_tools import (
    CpSolver,
    MinizincCpSolver,
    ParametersCp,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Problem,
)
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.lns_tools import (
    BaseLns,
    ConstraintHandler,
    InitialSolution,
    PostProcessSolution,
)
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
    fitness_class,
)

logger = logging.getLogger(__name__)


class BaseLnsCp(BaseLns):
    """Large Neighborhood Search solver using a cp solver at each iteration."""

    subsolver: CpSolver
    """Sub-solver used by this lns solver at each iteration."""

    def __init__(
        self,
        problem: Problem,
        subsolver: Optional[CpSolver] = None,
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

    def solve(
        self,
        nb_iteration_lns: int,
        parameters_cp: Optional[ParametersCp] = None,
        time_limit_subsolver: Optional[float] = 100.0,
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
            parameters_cp: parameters needed by the cp solver
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
        if parameters_cp is None:
            parameters_cp = ParametersCp.default()
        return super().solve(
            parameters_cp=parameters_cp,
            time_limit_subsolver=time_limit_subsolver,
            time_limit_subsolver_iter0=time_limit_subsolver_iter0,
            nb_iteration_lns=nb_iteration_lns,
            nb_iteration_no_improvement=nb_iteration_no_improvement,
            skip_initial_solution_provider=skip_initial_solution_provider,
            stop_first_iteration_if_optimal=stop_first_iteration_if_optimal,
            callbacks=callbacks,
            **kwargs,
        )


class LnsOrtoolsCpSat(BaseLnsCp):
    subsolver: OrtoolsCpSatSolver
    """Sub-solver used by this lns solver at each iteration."""

    def __init__(
        self,
        problem: Problem,
        subsolver: Optional[OrtoolsCpSatSolver] = None,
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
        if self.subsolver.cp_model is None:
            self.subsolver.init_model(**kwargs)
            if self.subsolver.cp_model is None:  # for mypy
                raise RuntimeError(
                    "subsolver cp_model must not be None after calling init_model()!"
                )


class OrtoolsCpSatConstraintHandler(ConstraintHandler):
    """Base class for constraint handler for solvers based on ortools"""

    @abstractmethod
    def adding_constraint_from_results_store(
        self, solver: OrtoolsCpSatSolver, result_storage: ResultStorage, **kwargs: Any
    ) -> Iterable[Constraint]:
        ...

    def remove_constraints_from_previous_iteration(
        self,
        solver: OrtoolsCpSatSolver,
        previous_constraints: Iterable[Constraint],
        **kwargs: Any,
    ) -> None:
        """Clear specified cpsat constraints."""
        for cstr in previous_constraints:
            cstr.proto.Clear()


class MznConstraintHandler(ConstraintHandler):
    @abstractmethod
    def adding_constraint_from_results_store(
        self,
        solver: MinizincCpSolver,
        child_instance: Instance,
        result_storage: ResultStorage,
        last_result_store: Optional[ResultStorage] = None,
        **kwargs: Any,
    ) -> Iterable[Any]:
        ...

    def remove_constraints_from_previous_iteration(
        self,
        solver: SolverDO,
        previous_constraints: Iterable[Any],
        **kwargs: Any,
    ) -> None:
        """Remove previous constraints.

        Nothing to do for minizinc solvers as the constraints were added only to the child instance.

        """
        pass


class LnsCpMzn(BaseLnsCp):

    subsolver: MinizincCpSolver
    """Sub-solver used by this lns solver at each iteration."""

    constraint_handler: MznConstraintHandler

    def __init__(
        self,
        problem: Problem,
        subsolver: Optional[MinizincCpSolver] = None,
        initial_solution_provider: Optional[InitialSolution] = None,
        constraint_handler: Optional[MznConstraintHandler] = None,
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
        if self.subsolver.instance is None:
            self.subsolver.init_model(**kwargs)
            if self.subsolver.instance is None:  # for mypy
                raise RuntimeError(
                    "CP model instance must not be None after calling init_model()!"
                )

    def create_submodel(self) -> contextlib.AbstractContextManager:
        """Create a branch of the current instance, wrapped in a context manager."""
        return self.subsolver.instance.branch()
