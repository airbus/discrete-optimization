"""Minimal API for a discrete-optimization solver."""

#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations  # see annotations as str

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

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

if TYPE_CHECKING:  # only for type checkers
    try:
        import optuna
    except ImportError:
        pass


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

    @classmethod
    def suggest_hyperparameter_value_with_optuna(
        cls, trial: optuna.trial.Trial, name: str, **kwargs
    ) -> Any:
        """Suggest hyperparameter value during an Optuna trial.

        This can be used during Optuna hyperparameters tuning.

        Args:
            trial: optuna trial during hyperparameters tuning
            name: name of the hyperparameter to choose
            **kwargs: options for optuna hyperparameter suggestions

        Returns:


        kwargs can be used to pass relevant arguments to
        - trial.suggest_float()
        - trial.suggest_int()
        - trial.suggest_categorical()

        For instance it can
        - add a low/high value if not existing for the hyperparameter
          or override it to narrow the search. (for float or int hyperparameters)
        - add a step or log argument (for float or int hyperparameters,
          see optuna.trial.Trial.suggest_float())
        - override choices for categorical or enum parameters to narrow the search

        """
        return cls.get_hyperparameter(name=name).suggest_with_optuna(
            trial=trial, **kwargs
        )

    @classmethod
    def suggest_hyperparameters_values_with_optuna(
        cls,
        trial: optuna.trial.Trial,
        names: Optional[List[str]] = None,
        kwargs_by_name: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[Any]:
        """Suggest hyperparameter value during an Optuna trial.

        Args:
            trial: optuna trial during hyperparameters tuning
            names: names of the hyperparameters to choose.
                By default, all hyperparameters will be suggested, ordered as in `self.hyperparameters`.
            kwargs_by_name: options for optuna hyperparameter suggestions, by hyperparameter name

        Returns:

        kwargs_by_name[some_name] will be passed as **kwargs to suggest_hyperparameter_value_with_optuna(name=some_name)

        """
        if names is None:
            names = cls.get_hyperparameters_names()
        if kwargs_by_name is None:
            kwargs_by_name = {}
        return [
            cls.suggest_hyperparameter_value_with_optuna(
                trial=trial, name=name, **kwargs_by_name.get(name, {})
            )
            for name in names
        ]

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
