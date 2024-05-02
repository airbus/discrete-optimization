#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations  # see annotations as str

import inspect
from collections import defaultdict
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    Hyperparameter,
    SubBrickKwargsHyperparameter,
)

if TYPE_CHECKING:  # only for type checkers
    try:
        import optuna
    except ImportError:
        pass


class Hyperparametrizable:
    """Base class for classes like SolverDO having (tunable) hyperparmeters.

    They have utility methods to
    - retrieve available hyperparameters
    - fill kwargs with default hyperparameters values
    - suggest hyperparameters by making use of optuna trials methods

    """

    hyperparameters: List[Hyperparameter] = []
    """Hyperparameters available for this solver.

    These hyperparameters are to be feed to **kwargs found in
        - __init__()
        - init_model() (when available)
        - solve()

    """

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
    def copy_and_update_hyperparameters(
        cls, names: Optional[List[str]] = None, **kwargs_by_name: Dict[str, Any]
    ) -> List[Hyperparameter]:
        """Copy hyperparameters definition of this class and update them with specified kwargs.

        This is useful to define hyperparameters for a child class
        for which only choices of the hyperparameter change for instance.

        Args:
            names: names of hyperparameters to copy. Default to all.
            **kwargs_by_name: for each hyperparameter specified by its name,
                the attributes to update. If a given hyperparameter name is not specified,
                the hyperparameter is copied without further update.

        Returns:

        """
        if names is None:
            names = cls.get_hyperparameters_names()
        kwargs_by_name = defaultdict(dict, kwargs_by_name)  # add missing names
        return [
            _copy_and_update_attributes(h, **kwargs_by_name[h.name])
            for h in cls.hyperparameters
            if h.name in names
        ]

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
            fixed_hyperparameters: values of fixed hyperparameters, useful for suggesting subbrick hyperparameters,
                if the subbrick class is not suggested by this method, but already fixed.
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
            if not (isinstance(name2hyperparameter[name], SubBrickKwargsHyperparameter))
        ]
        subsolvers_kwargs_hyperparameters: List[SubBrickKwargsHyperparameter] = [
            name2hyperparameter[name]
            for name in names
            if isinstance(name2hyperparameter[name], SubBrickKwargsHyperparameter)
        ]

        suggested_hyperparameters = {
            name: cls.suggest_hyperparameter_with_optuna(
                trial=trial, name=name, prefix=prefix, **kwargs_by_name.get(name, {})
            )
            for name in first_batch_hyperparameter_names
        }
        for hyperparameter in subsolvers_kwargs_hyperparameters:
            kwargs_for_optuna_suggestion = kwargs_by_name.get(hyperparameter.name, {})
            if hyperparameter.subbrick_hyperparameter in names:
                kwargs_for_optuna_suggestion["subbrick"] = suggested_hyperparameters[
                    hyperparameter.subbrick_hyperparameter
                ]
            elif hyperparameter.subbrick_hyperparameter in fixed_hyperparameters:
                kwargs_for_optuna_suggestion["subbrick"] = fixed_hyperparameters[
                    hyperparameter.subbrick_hyperparameter
                ]
            else:
                raise ValueError(
                    f"The choice of '{hyperparameter.subbrick_hyperparameter}' should be "
                    "either suggested by this method with `names` containing it or being None "
                    "or given via `fixed_hyperparameters`."
                )
            kwargs_for_optuna_suggestion[
                "prefix"
            ] = f"{prefix}{hyperparameter.subbrick_hyperparameter}."
            suggested_hyperparameters[
                hyperparameter.name
            ] = cls.suggest_hyperparameter_with_optuna(
                trial=trial, name=hyperparameter.name, **kwargs_for_optuna_suggestion
            )

        return suggested_hyperparameters


def _copy_and_update_attributes(h: Hyperparameter, **kwargs) -> Hyperparameter:
    hyperparameter_cls = h.__class__
    init_args_names = list(inspect.signature(h.__init__).parameters)
    for name in init_args_names:
        if name not in kwargs:
            kwargs[name] = getattr(h, name)
    h_new = hyperparameter_cls(**kwargs)
    return h_new
