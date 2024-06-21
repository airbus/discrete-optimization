#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations  # see annotations as str

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Container, Dict, Iterable, List
from typing import Mapping as MappingType
from typing import Optional, Tuple, Type, Union

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

    depends_on: Optional[Tuple[str, Container[Any]]] = None
    """Other hyperparameter on which this ones depends on.

    If None: this hyperparameter is always needed.
    Else:
        depends_on = hyperparameter2.name, possible_values
        this hyperparameter is needed if hyperparameter2 value is in possible_values.

    Warning: For now, the hyperparameter on which this one depends on cannot be a SubBrickKwargsHyperparameter.

    Notes:
        - How to define possible_values?
          - Usually a set or a list can be used. But sometime we need something smarter.
          - For integer or float hyperparameters, possible_values could be an interval (e.g. by using pandas.Interval)
        - For now, only simple dependency on a single hyperparameter, and a "set" of values is possible.
          The api could evolve to emcompass dependency on several other hyperparameters and more complex condition.

    """

    name_in_kwargs: Optional[str] = None
    """Corresponding key in kwargs when suggested via `solver.suggest_hyperparameters_with_optuna().`

    Default to hyperparemeter name. Can be used to have several hyperparameter with different limits/types
    depending on other hyperparameters value but supposed to share the same name in kwargs for solver initialization.
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

    step: int = 1
    """step to discretize."""

    log: bool = False
    """Whether to sample the value in a logarithmic scale."""

    default: Optional[int] = None
    """Default value for the hyperparameter.

    None means "no default value".

    """

    depends_on: Optional[Tuple[str, Container[Any]]] = None
    """Other hyperparameter on which this ones depends on.

    If None: this hyperparameter is always needed.
    Else:
        depends_on = hyperparameter2.name, possible_values
        this hyperparameter is needed if hyperparameter2 value is in possible_values.

    Warning: For now, the hyperparameter on which this one depends on cannot be a SubBrickKwargsHyperparameter.

    Notes:
        - How to define possible_values?
          - Usually a set or a list can be used. But sometime we need something smarter.
          - For integer or float hyperparameters, possible_values could be an interval (e.g. by using pandas.Interval)
        - For now, only simple dependency on a single hyperparameter, and a "set" of values is possible.
          The api could evolve to emcompass dependency on several other hyperparameters and more complex condition.

    """

    name_in_kwargs: Optional[str] = None
    """Corresponding key in kwargs when suggested via `solver.suggest_hyperparameters_with_optuna().`

    Default to hyperparemeter name. Can be used to have several hyperparameter with different limits/types
    depending on other hyperparameters value but supposed to share the same name in kwargs for solver initialization.
    """

    def suggest_with_optuna(
        self,
        trial: optuna.trial.Trial,
        low: Optional[int] = None,
        high: Optional[int] = None,
        step: Optional[int] = None,
        log: Optional[bool] = None,
        prefix: str = "",
        **kwargs: Any,
    ) -> Any:
        """Suggest hyperparameter value for an Optuna trial.

        Args:
            trial: optuna Trial used for choosing the hyperparameter value
            low: can be used to restrict lower bound
            high: can be used to restrict upper bound
            step: can be used to discretize by a given step
            log: whether to sample the value in a logarithmic scale
            prefix: prefix to add to optuna corresponding parameter name
              (useful for disambiguating hyperparameters from subsolvers in case of meta-solvers)
            **kwargs: passed to `trial.suggest_int()`

        Returns:

        """
        if low is None:
            low = self.low
        if high is None:
            high = self.high
        if step is None:
            step = self.step
        if log is None:
            log = self.log
        return trial.suggest_int(name=prefix + self.name, low=low, high=high, step=step, log=log, **kwargs)  # type: ignore


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

    step: Optional[float] = None
    """step to discretize if not None."""

    log: bool = False
    """Whether to sample the value in a logarithmic scale."""

    default: Optional[float] = None
    """Default value for the hyperparameter.

    None means "no default value".

    """

    depends_on: Optional[Tuple[str, Container[Any]]] = None
    """Other hyperparameter on which this ones depends on.

    If None: this hyperparameter is always needed.
    Else:
        depends_on = hyperparameter2.name, possible_values
        this hyperparameter is needed if hyperparameter2 value is in possible_values.

    Warning: For now, the hyperparameter on which this one depends on cannot be a SubBrickKwargsHyperparameter.

    Notes:
        - How to define possible_values?
          - Usually a set or a list can be used. But sometime we need something smarter.
          - For integer or float hyperparameters, possible_values could be an interval (e.g. by using pandas.Interval)
        - For now, only simple dependency on a single hyperparameter, and a "set" of values is possible.
          The api could evolve to emcompass dependency on several other hyperparameters and more complex condition.

    """

    name_in_kwargs: Optional[str] = None
    """Corresponding key in kwargs when suggested via `solver.suggest_hyperparameters_with_optuna().`

    Default to hyperparemeter name. Can be used to have several hyperparameter with different limits/types
    depending on other hyperparameters value but supposed to share the same name in kwargs for solver initialization.
    """

    def suggest_with_optuna(
        self,
        trial: optuna.trial.Trial,
        low: Optional[float] = None,
        high: Optional[float] = None,
        log: Optional[bool] = None,
        prefix: str = "",
        **kwargs: Any,
    ) -> Any:
        """Suggest hyperparameter value for an Optuna trial.

        Args:
            trial: optuna Trial used for choosing the hyperparameter value
            low: can be used to restrict lower bound
            high: can be used to restrict upper bound
            log: whether to sample the value in a logarithmic scale
            step: step of discretization if specified.
                If explicitely set to None, no discretization performed.
                By default, use self.step (and thus default discretization only if self.step not None)
            prefix: prefix to add to optuna corresponding parameter name
              (useful for disambiguating hyperparameters from subsolvers in case of meta-solvers)
            **kwargs: passed to `trial.suggest_float()`

        Returns:

        """
        if low is None:
            low = self.low
        if high is None:
            high = self.high
        if log is None:
            log = self.log
        if "step" in kwargs:
            step = kwargs.pop("step")
        else:
            step = self.step
        return trial.suggest_float(
            name=prefix + self.name, low=low, high=high, log=log, step=step, **kwargs  # type: ignore
        )


LabelType = Optional[Union[bool, int, float, str]]
"""Licit labels type for categorical hyperparameter."""


class CategoricalHyperparameter(Hyperparameter):
    """Categorical hyperparameter."""

    choices: MappingType[LabelType, Any]
    """Mapping lables to corresponding possible choices."""

    def __init__(
        self,
        name: str,
        choices: Union[Iterable[LabelType], MappingType[LabelType, Any]],
        default: Optional[Any] = None,
        depends_on: Optional[Tuple[str, Container[Any]]] = None,
        name_in_kwargs: Optional[str] = None,
    ):
        super().__init__(
            name=name,
            default=default,
            depends_on=depends_on,
            name_in_kwargs=name_in_kwargs,
        )
        if isinstance(choices, Mapping):
            self.choices = choices
        else:
            # list given instead of a mapping: choices should already be suitable labels
            self.choices = {c: c for c in choices}

    def suggest_with_optuna(
        self,
        trial: optuna.trial.Trial,
        choices: Optional[
            Union[Iterable[LabelType], MappingType[LabelType, Any]]
        ] = None,
        prefix: str = "",
        **kwargs: Any,
    ) -> Any:
        """Suggest hyperparameter value for an Optuna trial.

        Args:
            trial: optuna Trial used for choosing the hyperparameter value
            choices: restricts choices
            prefix: prefix to add to optuna corresponding parameter name
              (useful for disambiguating hyperparameters from subsolvers in case of meta-solvers)
            **kwargs: passed to `trial.suggest_categorical()`

        Returns:

        """
        if choices is None:
            choices = self.choices
        elif not isinstance(choices, Mapping):
            choices = {c: c for c in choices}

        label = trial.suggest_categorical(name=prefix + self.name, choices=choices.keys(), **kwargs)  # type: ignore
        return choices[label]


class EnumHyperparameter(CategoricalHyperparameter):
    """Hyperparameter taking value among an enumeration.

    Args:
        enum: enumeration used to create the hyperparameter
        choices: subset of the enumeration allowed. By default, the whole enumeration.

    """

    def __init__(
        self,
        name: str,
        enum: Type[Enum],
        choices: Optional[Union[Iterable[Enum], Dict[str, Enum]]] = None,
        default: Optional[Any] = None,
        depends_on: Optional[Tuple[str, Container[Any]]] = None,
        name_in_kwargs: Optional[str] = None,
    ):
        if choices is None:
            choices = {c.name: c for c in enum}
        elif not isinstance(choices, Mapping):
            choices = {c.name: c for c in choices}
        super().__init__(
            name,
            choices=choices,
            default=default,
            depends_on=depends_on,
            name_in_kwargs=name_in_kwargs,
        )
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
        else:
            choices = {c.name: c for c in choices}
        return super().suggest_with_optuna(
            trial=trial, choices=choices, prefix=prefix, **kwargs
        )


class SubBrickHyperparameter(CategoricalHyperparameter):
    """Hyperparameter whose values are Hyperparametrizable subclasses themselves.

    For instance subsolvers for meta-solvers.

    """

    choices: Dict[str, Type[Hyperparametrizable]]
    """Mapping of labelled Hyperparametrizable subclasses to choose from for the subbrick.

    NB: for now, it is not possible to pick the metasolver itself as a choice for its subbrick,
    in order to avoid infinite recursivity issues.

    """

    include_module_in_labels: bool
    """Flag to include module path in Hyperparametrizable labels used by optuna to select the value.

    This is useful if 2 hypermarametrizable classes share the same name but come from different modules.

    """

    def __init__(
        self,
        name: str,
        choices: Union[
            Dict[str, Type[Hyperparametrizable]], Iterable[Type[Hyperparametrizable]]
        ],
        default: Optional[Any] = None,
        depends_on: Optional[Tuple[str, Container[Any]]] = None,
        name_in_kwargs: Optional[str] = None,
        include_module_in_labels: bool = False,
    ):
        if not isinstance(choices, Mapping):
            if include_module_in_labels:
                choices = {c.__module__ + c.__name__: c for c in choices}
            else:
                choices = {c.__name__: c for c in choices}

        super().__init__(
            name,
            choices=choices,
            default=default,
            depends_on=depends_on,
            name_in_kwargs=name_in_kwargs,
        )
        self.include_module_in_labels = include_module_in_labels

    def suggest_with_optuna(
        self,
        trial: optuna.trial.Trial,
        choices: Optional[
            Union[
                Dict[str, Type[Hyperparametrizable]],
                Iterable[Type[Hyperparametrizable]],
            ]
        ] = None,
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
        elif not isinstance(choices, Mapping):
            if self.include_module_in_labels:
                choices = {c.__module__ + c.__name__: c for c in choices}
            else:
                choices = {c.__name__: c for c in choices}
        return super().suggest_with_optuna(
            trial=trial, choices=choices, prefix=prefix, **kwargs
        )


class SubBrickKwargsHyperparameter(Hyperparameter):
    """Keyword arguments for subbricks.

    This hyperparameter defines kwargs to be passed to the subbrick defined by another hyperparameter.

    Args:
        subbrick_hyperparameter: name of the SubBrickHyperparameter this hyperparameter corresponds to.
            If None, this means the subbrick is always constructed from the same class which should be then specified
            via `subbrick_cls`.
        subbrick_cls: class of the subbrick.
            Relevant only if `subbrick_hyperparmeter` is None. This means the class of the subbrick is always the same.

    """

    def __init__(
        self,
        name: str,
        subbrick_hyperparameter: Optional[str] = None,
        subbrick_cls: Optional[Type[Hyperparametrizable]] = None,
        default: Optional[Dict[str, Any]] = None,
        depends_on: Optional[Tuple[str, Container[Any]]] = None,
        name_in_kwargs: Optional[str] = None,
    ):
        super().__init__(
            name=name,
            default=default,
            depends_on=depends_on,
            name_in_kwargs=name_in_kwargs,
        )
        self.subbrick_cls = subbrick_cls
        self.subbrick_hyperparameter = subbrick_hyperparameter
        if subbrick_cls is None and subbrick_hyperparameter is None:
            raise ValueError(
                "subbricl_cls and subbrick_hyperparameter cannot be both None."
            )

    def suggest_with_optuna(
        self,
        trial: optuna.trial.Trial,
        subbrick: Optional[Type[Hyperparametrizable]] = None,
        names: Optional[List[str]] = None,
        kwargs_by_name: Optional[Dict[str, Dict[str, Any]]] = None,
        fixed_hyperparameters: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        """Suggest hyperparameter value for an Optuna trial.

        Args:
            trial: optuna Trial used for choosing the hyperparameter value
            subbrick: subbrick chosen as hyperparameter value for `self.subbrick_hyperparameter`.
                Can be None only if `self.subbrick_hyperparameter` is None and the subbrick class has already
                been specified by `self.subbrick_cls`.
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
        if subbrick is None:
            if self.subbrick_cls is None:
                raise ValueError(
                    "`subbrick` cannot be None if `self.subbrick_cls` is None."
                )
            else:
                subbrick = self.subbrick_cls
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
