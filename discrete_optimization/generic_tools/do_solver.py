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
    SubSolverKwargsHyperparameter,
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
    def get_default_hyperparameters(
        cls, names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get hyperparameters default values.

        Args:
            names: names of the hyperparameters to choose.
                By default, all available hyperparameters will be suggested.

        Returns:
            a mapping between hyperparameter name and its default value (None if not specified)

        """
        if names is None:
            names = cls.get_hyperparameters_names()
        hyperparameters_by_names = cls.get_hyperparameters_by_name()
        return {name: hyperparameters_by_names[name].default for name in names}

    @classmethod
    def complete_with_default_hyperparameters(
        cls, kwargs: Dict[str, Any], names: Optional[List[str]] = None
    ):
        """Add missing hyperparameters to kwargs by using default values

        Args:
            kwargs: keyword arguments to complete (for `__init__`, `init_model`, or `solve`)
            names: names of the hyperparameters to add if missing.
                By default, all available hyperparameters.

        Returns:
             a new dictionary, completion of kwargs

        """
        kwargs_complete = cls.get_default_hyperparameters(names=names)
        kwargs_complete.update(kwargs)  # ensure preferring values from kwargs
        return kwargs_complete

    @classmethod
    def suggest_hyperparameter_with_optuna(
        cls, trial: optuna.trial.Trial, name: str, prefix: str = "", **kwargs
    ) -> Any:
        """Suggest hyperparameter value during an Optuna trial.

        This can be used during Optuna hyperparameters tuning.

        Args:
            trial: optuna trial during hyperparameters tuning
            name: name of the hyperparameter to choose
            prefix: prefix to add to optuna corresponding parameter name
              (useful for disambiguating hyperparameters from subsolvers in case of meta-solvers)
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
            trial=trial, prefix=prefix, **kwargs
        )

    @classmethod
    def suggest_hyperparameters_with_optuna(
        cls,
        trial: optuna.trial.Trial,
        names: Optional[List[str]] = None,
        kwargs_by_name: Optional[Dict[str, Dict[str, Any]]] = None,
        fixed_hyperparameters: Optional[Dict[str, Any]] = None,
        prefix: str = "",
    ) -> Dict[str, Any]:
        """Suggest hyperparameters values during an Optuna trial.

        Args:
            trial: optuna trial during hyperparameters tuning
            names: names of the hyperparameters to choose.
                By default, all available hyperparameters will be suggested.
            kwargs_by_name: options for optuna hyperparameter suggestions, by hyperparameter name
            fixed_hyperparameters: values of fixed hyperparameters, useful for suggesting subsolver hyperparameters,
                if the subsolver class is not suggested by this method, but already fixed.
            prefix: prefix to add to optuna corresponding parameters
              (useful for disambiguating hyperparameters from subsolvers in case of meta-solvers)


        Returns:
            mapping between the hyperparameter name and its suggested value

        kwargs_by_name[some_name] will be passed as **kwargs to suggest_hyperparameter_with_optuna(name=some_name)

        """
        if names is None:
            names = cls.get_hyperparameters_names()
        if kwargs_by_name is None:
            kwargs_by_name = {}
        if fixed_hyperparameters is None:
            fixed_hyperparameters = {}

        # Meta-solvers: when defining subsolvers hyperparameters,
        #  be careful to suggest them before trying to suggest their own subset of hyperparameters
        name2hyperparameter = cls.get_hyperparameters_by_name()
        first_batch_hyperparameter_names = [
            name
            for name in names
            if not (
                isinstance(name2hyperparameter[name], SubSolverKwargsHyperparameter)
            )
        ]
        subsolvers_kwargs_hyperparameters: List[SubSolverKwargsHyperparameter] = [
            name2hyperparameter[name]
            for name in names
            if isinstance(name2hyperparameter[name], SubSolverKwargsHyperparameter)
        ]

        suggested_hyperparameters = {
            name: cls.suggest_hyperparameter_with_optuna(
                trial=trial, name=name, prefix=prefix, **kwargs_by_name.get(name, {})
            )
            for name in first_batch_hyperparameter_names
        }
        for hyperparameter in subsolvers_kwargs_hyperparameters:
            kwargs_for_optuna_suggestion = kwargs_by_name.get(hyperparameter.name, {})
            if hyperparameter.subsolver_hyperparameter in names:
                kwargs_for_optuna_suggestion["subsolver"] = suggested_hyperparameters[
                    hyperparameter.subsolver_hyperparameter
                ]
            elif hyperparameter.subsolver_hyperparameter in fixed_hyperparameters:
                kwargs_for_optuna_suggestion["subsolver"] = fixed_hyperparameters[
                    hyperparameter.subsolver_hyperparameter
                ]
            else:
                raise ValueError(
                    f"The choice of '{hyperparameter.subsolver_hyperparameter}' should be "
                    "either suggested by this method with `names` containing it or being None "
                    "or given via `fixed_hyperparameters`."
                )
            kwargs_for_optuna_suggestion[
                "prefix"
            ] = f"{prefix}{hyperparameter.subsolver_hyperparameter}."
            suggested_hyperparameters[
                hyperparameter.name
            ] = cls.suggest_hyperparameter_with_optuna(
                trial=trial, name=hyperparameter.name, **kwargs_for_optuna_suggestion
            )

        return suggested_hyperparameters

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
