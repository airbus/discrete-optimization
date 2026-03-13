#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import contextlib
from abc import abstractmethod
from typing import Any, Iterable, Optional

from minizinc import Instance

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Problem,
)
from discrete_optimization.generic_tools.lns_cp import BaseLnsCp
from discrete_optimization.generic_tools.lns_tools import (
    ConstraintHandler,
    InitialSolution,
    PostProcessSolution,
)
from discrete_optimization.generic_tools.mzn_tools import MinizincCpSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)


class MznConstraintHandler(ConstraintHandler):
    @abstractmethod
    def adding_constraint_from_results_store(
        self,
        solver: MinizincCpSolver,
        result_storage: ResultStorage,
        result_storage_last_iteration: ResultStorage,
        child_instance: Instance,
        **kwargs: Any,
    ) -> Iterable[Any]:
        """Add constraints to the internal model of a solver based on previous solutions

        Args:
            solver: solver whose internal model is updated
            result_storage: all results so far
            result_storage_last_iteration: results from last LNS iteration only
            child_instance: minizinc instance where to include the constraints
            **kwargs:

        Returns:
            empty list, not used

        """
        ...


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
