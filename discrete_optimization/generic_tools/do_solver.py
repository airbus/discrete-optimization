"""Minimal API for a discrete-optimization solver."""

#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from typing import Any, Dict, List, Optional

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Problem,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    Hyperparameter,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)


class SolverDO:
    """Base class for a discrete-optimization solver."""

    problem: Problem
    hyperparameters: List[Hyperparameter] = []
    """Hyperparameters available for this solver.

    These hyperparameters are to be feed to **kwargs found in
        - __init__()
        - init_model() (when available)
        - solve()

    """

    def __init__(
        self,
        problem: Problem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any
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

    @classmethod
    def get_hyperparameters_names(cls) -> List[str]:
        """List of hyperparameters names."""
        return [h.name for h in cls.hyperparameters]

    @classmethod
    def get_hyperparameters_by_name(cls) -> Dict[str, Hyperparameter]:
        """Mapping from name to corresponding hyperparameter."""
        return {h.name: h for h in cls.hyperparameters}

    @classmethod
    def get_hyperparameter(cls, name: str) -> Hyperparameter:
        """Get hyperparameter from given name."""
        return cls.get_hyperparameters_by_name()[name]

    @abstractmethod
    def solve(
        self, callbacks: Optional[List[Callback]] = None, **kwargs: Any
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

    def init_model(self, **kwargs: Any) -> None:
        """Initialize intern model used to solve.

        Can initialize a ortools, milp, gurobi, ... model.

        """
        pass
