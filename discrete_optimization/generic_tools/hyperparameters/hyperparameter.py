#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations  # see annotations as str

import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Container, Dict, Iterable, List
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
    """Corresponding key in generated kwargs.

    Used for kwargs generated by
    - `solver.suggest_hyperparameters_with_optuna()`
    - `solver.get_default_hyperparameters()`
    - `solver.complete_with_default_hyperparameters()`

    Default to hyperparemeter name. Can be used to have several hyperparameter with different limits/types
    depending on other hyperparameters value but supposed to share the same name in kwargs for solver initialization.
    """

    def __post_init__(self):
        if self.name_in_kwargs is None:
            self.name_in_kwargs = self.name

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

    def copy_and_update_attributes(self, **kwargs) -> Hyperparameter:
        hyperparameter_cls = self.__class__
        init_args_names = list(inspect.signature(self.__init__).parameters)
        for name in init_args_names:
            if name not in kwargs:
                kwargs[name] = getattr(self, name)
        h_new = hyperparameter_cls(**kwargs)
        return h_new


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

    suggest_low: bool = False
    """Whether to potentially suggest the lower bound.

    If step is None, optuna will suggest a float inside the range (low, high),
    but will never suggest exactly the lower bound by default.
    To force the behaviour, we will introduce a derived categorical hyperparameter
    whose name will be the hyperparameter name suffixed with ".suggest_bound".

    If step is not None, this attribute should probably be let to False.

    """

    suggest_high: bool = False
    """Whether to potentially suggest the upper bound.

    If step is None, optuna will suggest a float inside the range (low, high),
    but will never suggest exactly the upper bound by default.
    To force the behaviour, we will introduce a derived categorical hyperparameter
    whose name will be the hyperparameter name suffixed with ".suggest_bound".

    If step is not None, this attribute should probably be let to False.

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
        suggest_low: Optional[bool] = None,
        suggest_high: Optional[bool] = None,
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
            suggest_low: if set, will override `suggest_low` attribute. See its documentation.
            suggest_high: if set, will override `suggest_high` attribute. See its documentation.
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
        if suggest_low is None:
            suggest_low = self.suggest_low
        if suggest_high is None:
            suggest_high = self.suggest_high

        if suggest_low or suggest_high:
            choices = [""]
            if suggest_low:
                choices.append("low")
            if suggest_high:
                choices.append("high")
            suggest_bound = trial.suggest_categorical(
                name=prefix + self.name + ".suggest_bound", choices=choices
            )
            if suggest_bound == "low":
                high = low  # restrict range to a singleton {low}
            elif suggest_bound == "high":
                low = high  # restrict range to a singleton {high}

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
        # `Hyperparametrizable.copy_and_update_hyperparameters()` need that __init__ args are also attributes:
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


class SubBrickClsHyperparameter(CategoricalHyperparameter):
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
        default: Optional[Type[Hyperparametrizable]] = None,
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
        # `Hyperparametrizable.copy_and_update_hyperparameters()` need that __init__ args are also attributes:
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
        names_by_subbrick: Optional[Dict[Type[Hyperparametrizable], List[str]]] = None,
        kwargs_by_name: Optional[Dict[str, Dict[str, Any]]] = None,
        kwargs_by_name_by_subbrick: Optional[
            Dict[Type[Hyperparametrizable], Dict[str, Dict[str, Any]]]
        ] = None,
        fixed_hyperparameters: Optional[Dict[str, Any]] = None,
        fixed_hyperparameters_by_subbrick: Optional[
            Dict[Type[Hyperparametrizable], Dict[str, Any]]
        ] = None,
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
            names_by_subbrick: similar to `names` but depending on type of subbrick chosen.
                `names` will be extended by `names_by_subbrick[subbrick]` (if the key exists)
                where `subbrick` is either the argument of this function, or (if None) `self.subbrick_cls`.
            kwargs_by_name: options for optuna hyperparameter suggestions, by hyperparameter name.
                Passed to `subbrick.suggest_hyperparameters_with_optuna()`.
            kwargs_by_name_by_subbrick: same as `kwargs_by_name` but depending on type of subbrick chosen.
                `kwargs_by_name` will be updated by `kwargs_by_name_by_subbrick[subbrick]` (if the key exists)
                where `subbrick` is either the argument of this function, or (if None) `self.subbrick_cls`.
            fixed_hyperparameters: values of fixed hyperparameters, useful for suggesting subbrick hyperparameters,
                if the subbrick class is not suggested by this method, but already fixed.
            fixed_hyperparameters_by_subbrick: same as `fixed_hyperparameters` but depending on type of subbrick chosen.
                `fixed_hyperparameters` will be updated by `fixed_hyperparameters_by_subbrick[subbrick]` (if the key exists)
                where `subbrick` is either the argument of this function, or (if None) `self.subbrick_cls`.
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

        # kwargs_by_name updated according to subbrick
        if kwargs_by_name is None:
            kwargs_by_name_updated = {}
        else:
            kwargs_by_name_updated = dict(kwargs_by_name)
        if kwargs_by_name_by_subbrick is not None:
            kwargs_by_name_updated.update(kwargs_by_name_by_subbrick.get(subbrick, {}))

        # fixed_hyperparameters updated according to subbrick
        if fixed_hyperparameters is None:
            fixed_hyperparameters_updated = {}
        else:
            fixed_hyperparameters_updated = dict(fixed_hyperparameters)
        if fixed_hyperparameters_by_subbrick is not None:
            fixed_hyperparameters_updated.update(
                fixed_hyperparameters_by_subbrick.get(subbrick, {})
            )

        # names updated according to subbrick
        subbrick_hyperparameter_names = subbrick.get_hyperparameters_names()
        if names is None:
            names_updated = list(subbrick_hyperparameter_names)
        else:
            names_updated = [
                name for name in names if name in subbrick_hyperparameter_names
            ]
        if names_by_subbrick is not None:
            names_updated.extend(names_by_subbrick.get(subbrick, []))

        # update prefix with subbrick name and class (if not fixed class)
        if self.subbrick_hyperparameter is None:
            prefix = f"{prefix}{self.name}."
        else:
            prefix = f"{prefix}{self.subbrick_hyperparameter}.{subbrick.__name__}."

        # use subbrick suggest method
        return subbrick.suggest_hyperparameters_with_optuna(
            trial=trial,
            names=names_updated,
            kwargs_by_name=kwargs_by_name_updated,
            fixed_hyperparameters=fixed_hyperparameters_updated,
            prefix=prefix,
            **kwargs,  # type: ignore
        )


class SubBrickHyperparameter(Hyperparameter):
    """Hyperparameter whose values are SubBrick instances.

    That is to say
        - a hyperparametrizable class
        - a kwargs dict to be used for it in __init_(), init_model(), solve(), ...


    This is useful to suggest subsolvers for meta-solvers.

    Under the hood, this hyperparameter will generate the corresponding SubBrickClsHyperparameter and
    SubBrickKwargsHyperparameter.

    """

    def __init__(
        self,
        name: str,
        choices: Union[
            Dict[str, Type[Hyperparametrizable]], Iterable[Type[Hyperparametrizable]]
        ],
        default: Optional[SubBrick] = None,
        depends_on: Optional[Tuple[str, Container[Any]]] = None,
        name_in_kwargs: Optional[str] = None,
        include_module_in_labels: bool = False,
    ):
        """

        Args:
            name: see Hyperparameter doc.
            choices: see SubBrickClsHyperparameter doc.
            default: see Hyperparameter doc.
            depends_on: see Hyperparameter doc.
            name_in_kwargs: see Hyperparameter doc.
            include_module_in_labels: See SubBrickClsHyperparameter doc.

        """
        super().__init__(
            name=name,
            default=default,
            depends_on=depends_on,
            name_in_kwargs=name_in_kwargs,
        )
        self.subbrick_cls_hp = SubBrickClsHyperparameter(
            name=f"{name}.cls",
            choices=choices,
            include_module_in_labels=include_module_in_labels,
        )
        self.subbrick_kwargs_hp = SubBrickKwargsHyperparameter(
            name=f"{name}.kwargs",
            subbrick_hyperparameter=name,
        )
        # `Hyperparametrizable.copy_and_update_hyperparameters()` need that __init__ args are also attributes:
        self.choices = self.subbrick_cls_hp.choices
        self.include_module_in_labels = self.subbrick_cls_hp.include_module_in_labels

    def suggest_with_optuna(
        self,
        trial: optuna.trial.Trial,
        choices: Optional[
            Union[
                Dict[str, Type[Hyperparametrizable]],
                Iterable[Type[Hyperparametrizable]],
            ]
        ] = None,
        names: Optional[List[str]] = None,
        names_by_subbrick: Optional[Dict[Type[Hyperparametrizable], List[str]]] = None,
        kwargs_by_name: Optional[Dict[str, Dict[str, Any]]] = None,
        kwargs_by_name_by_subbrick: Optional[
            Dict[Type[Hyperparametrizable], Dict[str, Dict[str, Any]]]
        ] = None,
        fixed_hyperparameters: Optional[Dict[str, Any]] = None,
        fixed_hyperparameters_by_subbrick: Optional[
            Dict[Type[Hyperparametrizable], Dict[str, Any]]
        ] = None,
        prefix: str = "",
        **kwargs: Any,
    ) -> SubBrick:
        """

        Args:
            trial: see Hyperparameter doc
            choices: used by underlying SubBrickClsHyperparameter.suggest_with_optuna
            names: used by underlying SubBrickKwargsHyperparameter.suggest_with_optuna
            names_by_subbrick: used by underlying SubBrickKwargsHyperparameter.suggest_with_optuna
            kwargs_by_name: used by underlying SubBrickKwargsHyperparameter.suggest_with_optuna
            kwargs_by_name_by_subbrick: used by underlying SubBrickKwargsHyperparameter.suggest_with_optuna
            fixed_hyperparameters: used by underlying SubBrickKwargsHyperparameter.suggest_with_optuna
            fixed_hyperparameters_by_subbrick: used by underlying SubBrickKwargsHyperparameter.suggest_with_optuna
            prefix: see Hyperparameter doc.
            **kwargs: passed to SubBrickClsHyperparameter.suggest_with_optuna
                and SubBrickKwargsHyperparameter.suggest_with_optuna

        Returns:

        """
        subbrick_cls = self.subbrick_cls_hp.suggest_with_optuna(
            trial=trial, choices=choices, prefix=prefix, **kwargs
        )
        subbrick_kwargs = self.subbrick_kwargs_hp.suggest_with_optuna(
            trial=trial,
            subbrick=subbrick_cls,
            names=names,
            names_by_subbrick=names_by_subbrick,
            kwargs_by_name=kwargs_by_name,
            kwargs_by_name_by_subbrick=kwargs_by_name_by_subbrick,
            fixed_hyperparameters=fixed_hyperparameters,
            fixed_hyperparameters_by_subbrick=fixed_hyperparameters_by_subbrick,
            prefix=prefix,
            **kwargs,
        )
        return SubBrick(cls=subbrick_cls, kwargs=subbrick_kwargs)


@dataclass
class SubBrick:
    """Wrapper class for a hyperparametrizable class and its kwargs.

    Meant to be used as output by `SubBrickHyperparameter.suggest_with_optuna()`.

    """

    cls: Type[Hyperparametrizable]
    kwargs: Dict[str, Any]
    kwargs_from_solution: Optional[Dict[str, Callable[..., Any]]] = None


class ListHyperparameter(Hyperparameter):
    """Variable list of hyperparameters.

    This represents a list of hyperparameters that are copies of a given template, with a bounded variable length.

    """

    hyperparameter_template: Hyperparameter
    """Hyperparameter template to fill the list."""

    length_high: int
    """Upper bound on list length."""

    length_low: int = 0
    "Lower bound for list length."

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
    """Corresponding key in generated kwargs.

    Used for kwargs generated by
    - `solver.suggest_hyperparameters_with_optuna()`
    - `solver.get_default_hyperparameters()`
    - `solver.complete_with_default_hyperparameters()`

    Default to hyperparemeter name. Can be used to have several hyperparameter with different limits/types
    depending on other hyperparameters value but supposed to share the same name in kwargs for solver initialization.
    """

    def __init__(
        self,
        name: str,
        hyperparameter_template: Hyperparameter,
        length_high: int,
        length_low: int = 0,
        default: List[Any] = None,
        depends_on: Optional[Tuple[str, Container[Any]]] = None,
        name_in_kwargs: Optional[str] = None,
    ):
        super().__init__(
            name=name,
            default=default,
            depends_on=depends_on,
            name_in_kwargs=name_in_kwargs,
        )
        self.hyperparameter_template = hyperparameter_template
        self.length_low = length_low
        self.length_high = length_high

    def suggest_with_optuna(
        self, trial: optuna.trial.Trial, prefix: str = "", **kwargs: Any
    ) -> List[Any]:
        """Suggest hyperparameter value for an Optuna trial.

        Args:
            trial: optuna Trial used for choosing the hyperparameter value
            prefix: prefix to add to optuna corresponding parameter name
              (useful for disambiguating hyperparameters from subsolvers in case of meta-solvers)
            **kwargs: passed to `trial.suggest_xxx()`

        Returns:

        """
        list_length_name = prefix + self.name + ".length"
        list_length = trial.suggest_int(
            name=list_length_name, low=self.length_low, high=self.length_high
        )
        list_hp = []
        for i in range(list_length):
            hp = self.hyperparameter_template.copy_and_update_attributes(
                name=f"{self.hyperparameter_template.name}_{i}"
            )
            list_hp.append(hp.suggest_with_optuna(trial=trial, prefix=prefix, **kwargs))
        return list_hp
