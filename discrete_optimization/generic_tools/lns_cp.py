#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import contextlib
import logging
import random
import sys
from abc import abstractmethod
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from minizinc import Instance
from ortools.sat.python.cp_model import Constraint

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.cp_tools import (
    CPSolver,
    MinizincCPSolver,
    ParametersCP,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Problem,
)
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.lns_tools import (
    BaseLNS,
    ConstraintHandler,
    InitialSolution,
    PostProcessSolution,
)
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCPSatSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
    fitness_class,
)

if sys.version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict


logger = logging.getLogger(__name__)


class BaseLNS_CP(BaseLNS):
    """Large Neighborhood Search solver using a cp solver at each iteration."""

    subsolver: CPSolver
    """Sub-solver used by this lns solver at each iteration."""

    def __init__(
        self,
        problem: Problem,
        subsolver: Optional[CPSolver] = None,
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

    def solve_with_subsolver(
        self,
        no_previous_solution: bool,
        instance: Instance,
        parameters_cp: ParametersCP,
        **kwargs: Any,
    ) -> ResultStorage:
        if no_previous_solution:
            parameters_cp0 = parameters_cp.copy()
            parameters_cp0.time_limit = parameters_cp.time_limit_iter0
            result_store = self.subsolver.solve(
                parameters_cp=parameters_cp0, instance=instance
            )
        else:
            result_store = self.subsolver.solve(
                parameters_cp=parameters_cp, instance=instance
            )
        return result_store

    def solve(
        self,
        nb_iteration_lns: int,
        parameters_cp: Optional[ParametersCP] = None,
        nb_iteration_no_improvement: Optional[int] = None,
        skip_initial_solution_provider: bool = False,
        stop_first_iteration_if_optimal: bool = True,
        callbacks: Optional[List[Callback]] = None,
        **kwargs: Any,
    ) -> ResultStorage:
        if parameters_cp is None:
            parameters_cp = ParametersCP.default()
        return super().solve(
            parameters_cp=parameters_cp,
            nb_iteration_lns=nb_iteration_lns,
            nb_iteration_no_improvement=nb_iteration_no_improvement,
            skip_initial_solution_provider=skip_initial_solution_provider,
            stop_first_iteration_if_optimal=stop_first_iteration_if_optimal,
            callbacks=callbacks,
            **kwargs,
        )


class LNS_OrtoolsCPSat(BaseLNS_CP):
    subsolver: OrtoolsCPSatSolver
    """Sub-solver used by this lns solver at each iteration."""

    def __init__(
        self,
        problem: Problem,
        subsolver: Optional[OrtoolsCPSatSolver] = None,
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


class OrtoolsCPSatConstraintHandler(ConstraintHandler):
    """Base class for constraint handler for solvers based on ortools"""

    @abstractmethod
    def adding_constraint_from_results_store(
        self, solver: OrtoolsCPSatSolver, result_storage: ResultStorage, **kwargs: Any
    ) -> Iterable[Constraint]:
        ...

    def remove_constraints_from_previous_iteration(
        self,
        solver: OrtoolsCPSatSolver,
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
        solver: MinizincCPSolver,
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


class LNS_MinizincCP(BaseLNS_CP):

    subsolver: MinizincCPSolver
    """Sub-solver used by this lns solver at each iteration."""

    constraint_handler: MznConstraintHandler

    def __init__(
        self,
        problem: Problem,
        subsolver: Optional[MinizincCPSolver] = None,
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


# Alias for backward compatibility
LNS_CP = LNS_MinizincCP
