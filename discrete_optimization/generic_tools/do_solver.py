"""Minimal API for a discrete-optimization solver."""

#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations  # see annotations as str

from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import Enum
from typing import Any, Optional

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Problem,
    Solution,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparametrizable import (
    Hyperparametrizable,
)
from discrete_optimization.generic_tools.result_storage.multiobj_utils import (
    TupleFitness,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
    fitness_class,
)


class StatusSolver(Enum):
    SATISFIED = "SATISFIED"
    UNSATISFIABLE = "UNSATISFIABLE"
    OPTIMAL = "OPTIMAL"
    UNKNOWN = "UNKNOWN"
    ERROR = "FAILED"


class SolverDO(Hyperparametrizable, ABC):
    """Base class for a discrete-optimization solver."""

    problem: Problem
    status_solver: StatusSolver = StatusSolver.UNKNOWN

    def __init__(
        self,
        problem: Problem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        self.problem = problem
        (
            self.aggreg_from_sol,
            self.aggreg_from_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.problem,
            params_objective_function=params_objective_function,
        )

    @abstractmethod
    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        """Generic solving function.

        Args:
            callbacks: list of callbacks used to hook into the various stage of the solve
            **kwargs: any argument specific to the solver

        Solvers deriving from SolverDo should use callbacks methods .on_step_end(), ...
        during solve(). But some solvers are not yet updated and are just ignoring it.

        Returns (ResultStorage): a result object containing potentially a pool of solutions
        to a discrete-optimization problem
        """
        ...

    def create_result_storage(
        self, list_solution_fits: Optional[list[tuple[Solution, fitness_class]]] = None
    ) -> ResultStorage:
        """Create a result storage with the proper mode_optim.

        Args:
            list_solution_fits:

        Returns:

        """
        if list_solution_fits is None:
            list_solution_fits = []
        return ResultStorage(
            list_solution_fits=list_solution_fits,
            mode_optim=self.params_objective_function.sense_function,
        )

    def init_model(self, **kwargs: Any) -> None:
        """Initialize internal model used to solve.

        Can initialize a ortools, milp, gurobi, ... model.

        """
        ...

    def is_optimal(self) -> Optional[bool]:
        """Tell if found solution is supposed to be optimal.

        To be called after a solve.

        Returns:
            optimality of the solution. If information missing, returns None instead.

        """
        if self.status_solver == StatusSolver.UNKNOWN:
            return None
        else:
            return self.status_solver == StatusSolver.OPTIMAL

    def get_lexico_objectives_available(self) -> list[str]:
        """List objectives available for lexico optimization

        It corresponds to the labels accepted for obj argument for
        - `set_lexico_objective()`
        - `add_lexico_constraint()`
        - `get_lexico_objective_value()`

        Default to `self.problem.get_objective_names()`.

        Returns:

        """
        return self.problem.get_objective_names()

    def set_lexico_objective(self, obj: str) -> None:
        """Update internal model objective.

        Args:
            obj: a string representing the desired objective.
                Should be one of `self.get_lexico_objectives_available()`.

        Returns:

        """
        ...

    def get_lexico_objective_value(self, obj: str, res: ResultStorage) -> float:
        """Get best internal model objective value found by last call to `solve()`.

        The default implementation consists in using the fit of the last solution in result_storage.
        This assumes:
        - that the last solution is the best one for the objective considered
        - that no aggregation was performed but rather that the fitness is a TupleFitness
          with values in the same order as `self.problem.get_objective_names()`.

        Args:
            obj: a string representing the desired objective.
                Should be one of `self.get_lexico_objectives_available()`.
            res: result storage returned by last call to solve().

        Returns:

        """
        _, fit = res[-1]
        if not isinstance(fit, TupleFitness):
            raise RuntimeError(
                "The fitness should be a TupleFitness of the same size as `self.problem.get_objective_names()`."
            )
        objectives = self.problem.get_objective_names()
        idx = objectives.index(obj)
        return float(fit.vector_fitness[idx])

    def add_lexico_constraint(self, obj: str, value: float) -> Iterable[Any]:
        """Add a constraint on a computed sub-objective

        Args:
            obj: a string representing the desired objective.
                Should be one of `self.get_lexico_objectives_available()`.
            value: the limiting value.
                If the optimization direction is maximizing, this is a lower bound,
                else this is an upper bound.

        Returns:
            the created constraints.

        """
        ...

    def remove_constraints(self, constraints: Iterable[Any]) -> None:
        """Remove the internal model constraints.

        Args:
            constraints: constraints created for instance with `add_lexico_constraint()`

        Returns:

        """
        ...

    @staticmethod
    def implements_lexico_api() -> bool:
        """Tell whether this solver is implementing the api for lexicographic optimization.

        Should return True only if

        - `set_lexico_objective()`
        - `add_lexico_constraint()`
        - `get_lexico_objective_value()`

        have been really implemented, i.e.
        - calling `set_lexico_objective()` and `add_lexico_constraint()`
          should actually change the next call to `solve()`,
        - `get_lexico_objective_value()` should correspond to the internal model objective

        """
        return False


class WarmstartMixin(ABC):
    """Mixin class for warmstart-ready solvers."""

    @abstractmethod
    def set_warm_start(self, solution: Solution) -> None:
        """Make the solver warm start from the given solution."""
        ...


class BoundsProviderMixin(ABC):
    """Mixin class for solvers providing bounds on objective.

    e.g. to be used in callback for early stopping.

    """

    @abstractmethod
    def get_current_best_internal_objective_bound(self) -> Optional[float]:
        ...

    @abstractmethod
    def get_current_best_internal_objective_value(self) -> Optional[float]:
        ...

    def get_current_absolute_gap(self) -> Optional[float]:
        """Get current (absolute) optimality gap.

        i.e. |current_obj_value - current_obj_bound|

        """
        bound = self.get_current_best_internal_objective_bound()
        best_sol = self.get_current_best_internal_objective_value()
        if bound is None or best_sol is None:
            return None
        else:
            return abs(best_sol - bound)


class TrivialSolverFromResultStorage(SolverDO, WarmstartMixin):
    """Trivial solver created from an already computed result storage."""

    def __init__(
        self,
        problem: Problem,
        result_storage: ResultStorage,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(
            problem=problem,
            params_objective_function=params_objective_function,
            **kwargs,
        )
        self.result_storage = result_storage

    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        return self.result_storage

    def set_warm_start(self, solution: Solution) -> None:
        """Add solution add result_storage's start."""
        fit = self.aggreg_from_sol(solution)
        self.result_storage.insert(0, (solution, fit))


class TrivialSolverFromSolution(SolverDO):
    """Trivial solver created from an already computed solution."""

    def __init__(
        self,
        problem: Problem,
        solution: Solution,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(
            problem=problem,
            params_objective_function=params_objective_function,
            **kwargs,
        )
        self.set_warm_start(solution)

    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        return self.result_storage

    def set_warm_start(self, solution: Solution) -> None:
        """Replace the stored solution."""
        fit = self.aggreg_from_sol(solution)
        self.result_storage = self.create_result_storage([(solution, fit)])
