#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations  # see annotations as str

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Type

if TYPE_CHECKING:  # only for type checkers
    from discrete_optimization.generic_tools.hyperparameters.hyperparametrizable import (
        Hyperparametrizable,
    )

    try:
        import optuna  # not necessary to import this module
    except ImportError:
        pass


@dataclass
class Hyperparameter:
    """Hyperparameter base class used to specify d-o solver hyperparameters."""

    name: str
    """Name of the hyperparameter.

    Should correspond to how the hyperparameter is specified in the solver
    `__init__()`, `init_model()`, or `solve()` keywords arguments.

    """

    default: Optional[Any] = None
    """Default value for the hyperparameter.

    None means "no default value".

    """

    def suggest_with_optuna(
        self, trial: optuna.trial.Trial, prefix: str = "", **kwargs: Any
    ) -> Any:
        """Suggest hyperparameter value for an Optuna trial.

        Args:
            trial: optuna Trial used for choosing the hyperparameter value
            prefix: prefix to add to optuna corresponding parameter name
              (useful for disambiguating hyperparameters from subsolvers in case of meta-solvers)
            **kwargs: passed to `trial.suggest_xxx()`

        Returns:

        """
        ...


@dataclass
class IntegerHyperparameter(Hyperparameter):
    """Integer hyperparameter."""

    low: Optional[int] = None
    """Lower bound.

    If None, the hyperparameter value has no lower bound.

    """

    high: Optional[int] = None
    """Upper bound.

    If None, the hyperparameter value has no upper bound.

    """

    default: Optional[int] = None
    """Default value for the hyperparameter.

    None means "no default value".

    """

    def suggest_with_optuna(
        self,
        trial: optuna.trial.Trial,
        low: Optional[int] = None,
        high: Optional[int] = None,
        prefix: str = "",
        **kwargs: Any,
    ) -> Any:
        """Suggest hyperparameter value for an Optuna trial.

        Args:
            trial: optuna Trial used for choosing the hyperparameter value
            low: can be used to restrict lower bound
            high: can be used to restrict upper bound
            prefix: prefix to add to optuna corresponding parameter name
              (useful for disambiguating hyperparameters from subsolvers in case of meta-solvers)
            **kwargs: passed to `trial.suggest_int()`

        Returns:

        """
        if low is None:
            low = self.low
        if high is None:
            high = self.high
        return trial.suggest_int(name=prefix + self.name, low=low, high=high, **kwargs)  # type: ignore


@dataclass
class FloatHyperparameter(Hyperparameter):
    """Float parameter."""

    low: Optional[float] = None
    """Lower bound.

    If None, the hyperparameter value has no lower bound.

    """

    high: Optional[float] = None
    """Upper bound.

    If None, the hyperparameter value has no upper bound.

    """

    default: Optional[float] = None
    """Default value for the hyperparameter.

    None means "no default value".

    """

    def suggest_with_optuna(
        self,
        trial: optuna.trial.Trial,
        low: Optional[float] = None,
        high: Optional[float] = None,
        prefix: str = "",
        **kwargs: Any,
    ) -> Any:
        """Suggest hyperparameter value for an Optuna trial.

        Args:
            trial: optuna Trial used for choosing the hyperparameter value
            low: can be used to restrict lower bound
            high: can be used to restrict upper bound
            prefix: prefix to add to optuna corresponding parameter name
              (useful for disambiguating hyperparameters from subsolvers in case of meta-solvers)
            **kwargs: passed to `trial.suggest_float()`

        Returns:

        """
        if low is None:
            low = self.low
        if high is None:
            high = self.high
        return trial.suggest_float(
            name=prefix + self.name, low=low, high=high, **kwargs
        )


@dataclass
class CategoricalHyperparameter(Hyperparameter):
    """Categorical hyperparameter."""

    choices: List[Any] = field(default_factory=list)
    """List of possible choices."""

    def suggest_with_optuna(
        self,
        trial: optuna.trial.Trial,
        choices: Optional[Iterable[Any]] = None,
        prefix: str = "",
        **kwargs: Any,
    ) -> Any:
        """Suggest hyperparameter value for an Optuna trial.

        Args:
            trial: optuna Trial used for choosing the hyperparameter value
            choices: restricts list of choices
            prefix: prefix to add to optuna corresponding parameter name
              (useful for disambiguating hyperparameters from subsolvers in case of meta-solvers)
            **kwargs: passed to `trial.suggest_categorical()`

        Returns:

        """
        if choices is None:
            choices = self.choices

        return trial.suggest_categorical(name=prefix + self.name, choices=choices, **kwargs)  # type: ignore


class EnumHyperparameter(CategoricalHyperparameter):
    """Hyperparameter taking value among an enumeration.

    Args:
        enum: enumeration used to create the hyperparameter

    """

    def __init__(self, name: str, enum: Type[Enum], default: Optional[Any] = None):
        super().__init__(name, choices=list(enum), default=default)
        self.enum = enum

    def suggest_with_optuna(
        self,
        trial: optuna.trial.Trial,
        choices: Optional[Iterable[Enum]] = None,
        prefix: str = "",
        **kwargs: Any,
    ) -> Enum:
        """Suggest hyperparameter value for an Optuna trial.

        Args:
            trial: optuna Trial used for choosing the hyperparameter value
            choices: restricts list of choices among the enumeration `self.enum`
            prefix: prefix to add to optuna corresponding parameter name
              (useful for disambiguating hyperparameters from subsolvers in case of meta-solvers)
            **kwargs: passed to `trial.suggest_categorical()`

        Returns:

        """
        if choices is None:
            choices = self.choices
        choices_str = [c.name for c in choices]
        choice_str: str = trial.suggest_categorical(name=prefix + self.name, choices=choices_str, **kwargs)  # type: ignore
        return self.enum[choice_str]


class SubBrickHyperparameter(CategoricalHyperparameter):
    """Hyperparameter whose values are Hyperparametrizable subclasses themselves.

    For instance subsolvers for meta-solvers.

    """

    choices: List[Type[Hyperparametrizable]]
    """List of Hyperparametrizable subclasses to choose from for the subbrick.

    NB: for now, it is not possible to pick the metasolver itself as a choice for its subbrick,
    in order to avoid infinite recursivity issues.

    """

    def __init__(
        self,
        name: str,
        choices: List[Type[Hyperparametrizable]],
        default: Optional[Any] = None,
    ):
        super().__init__(name, choices=choices, default=default)
        # map by their names or (module + name)'s?
        if len(set([c.__name__ for c in choices])) == len(choices):
            # names are unique between all choices
            self.choices_str2cls = {c.__name__: c for c in choices}
            self.choices_cls2str = {c: c.__name__ for c in choices}
        else:
            # we need to disambiguate with module path
            self.choices_str2cls = {c.__module__ + c.__name__: c for c in choices}
            self.choices_cls2str = {c: c.__module__ + c.__name__ for c in choices}

    def suggest_with_optuna(
        self,
        trial: optuna.trial.Trial,
        choices: Optional[Iterable[Type[Hyperparametrizable]]] = None,
        prefix: str = "",
        **kwargs: Any,
    ) -> Type[Hyperparametrizable]:
        """Suggest hyperparameter value for an Optuna trial.

        Args:
            trial: optuna Trial used for choosing the hyperparameter value
            choices: restricts list of subbricks to choose from
            prefix: prefix to add to optuna corresponding parameter name
              (useful for disambiguating hyperparameters from subsolvers in case of meta-solvers)
            **kwargs: passed to `trial.suggest_categorical()`

        Returns:

        """
        if choices is None:
            choices = self.choices
        choices_str = [self.choices_cls2str[c] for c in choices]
        choice_str = trial.suggest_categorical(
            name=prefix + self.name, choices=choices_str, **kwargs  # type: ignore
        )
        return self.choices_str2cls[choice_str]


class SubBrickKwargsHyperparameter(Hyperparameter):
    """Keyword arguments for subbricks.

    This hyperparameter defines kwargs to be passed to the subbrick defined by another hyperparameter.

    Args:
        subbrick_hyperparameter: name of the SubBrickHyperparameter this hyperparameter corresponds to.

    """

    def __init__(
        self,
        name: str,
        subbrick_hyperparameter: str,
        default: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, default=default)
        self.subbrick_hyperparameter = subbrick_hyperparameter

    def suggest_with_optuna(
        self,
        trial: optuna.trial.Trial,
        subbrick: Type[Hyperparametrizable],
        names: Optional[List[str]] = None,
        kwargs_by_name: Optional[Dict[str, Dict[str, Any]]] = None,
        fixed_hyperparameters: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        """Suggest hyperparameter value for an Optuna trial.

        Args:
            trial: optuna Trial used for choosing the hyperparameter value
            subbrick: subbrick chosen as hyperparameter value for `self.subbrick_hyperparameter`
            names: names of the hyperparameters to choose for the subbrick.
                Only relevant names will be considered (i.e. corresponding to existing hyperparameters names for the chosen subbrick),
                the other will be discarded (potentially, being meaningful for other subbricks).
                By default, all available hyperparameters will be suggested.
                Passed to `subbrick.suggest_hyperparameters_with_optuna()`.
            kwargs_by_name: options for optuna hyperparameter suggestions, by hyperparameter name.
                Passed to `subbrick.suggest_hyperparameters_with_optuna()`.
            fixed_hyperparameters: values of fixed hyperparameters, useful for suggesting subbrick hyperparameters,
                if the subbrick class is not suggested by this method, but already fixed.
            prefix: prefix to add to optuna corresponding parameter name
              (useful for disambiguating hyperparameters from subsolvers in case of meta-solvers)
            **kwargs: passed to `trial.suggest_categorical()`

        Returns:

        """
        subbrick_hyperparameter_names = subbrick.get_hyperparameters_names()
        if names is not None:
            names = [name for name in names if name in subbrick_hyperparameter_names]
        return subbrick.suggest_hyperparameters_with_optuna(
            trial=trial,
            names=names,
            kwargs_by_name=kwargs_by_name,
            fixed_hyperparameters=fixed_hyperparameters,
            prefix=prefix,
            **kwargs,  # type: ignore
        )
