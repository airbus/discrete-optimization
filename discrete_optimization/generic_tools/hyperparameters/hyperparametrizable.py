#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations  # see annotations as str

from collections import ChainMap, defaultdict
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Optional

import networkx as nx

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

    hyperparameters: list[Hyperparameter] = []
    """Hyperparameters available for this solver.

    These hyperparameters are to be feed to **kwargs found in
        - __init__()
        - init_model() (when available)
        - solve()

    """

    @classmethod
    def get_hyperparameters_names(cls) -> list[str]:
        """List of hyperparameters names."""
        return [h.name for h in cls.hyperparameters]

    @classmethod
    def get_hyperparameters_by_name(cls) -> dict[str, Hyperparameter]:
        """Mapping from name to corresponding hyperparameter."""
        return {h.name: h for h in cls.hyperparameters}

    @classmethod
    def get_hyperparameter(cls, name: str) -> Hyperparameter:
        """Get hyperparameter from given name."""
        return cls.get_hyperparameters_by_name()[name]

    @classmethod
    def copy_and_update_hyperparameters(
        cls, names: Optional[list[str]] = None, **kwargs_by_name: dict[str, Any]
    ) -> list[Hyperparameter]:
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
            h.copy_and_update_attributes(**kwargs_by_name[h.name])
            for h in cls.hyperparameters
            if h.name in names
        ]

    @classmethod
    def get_default_hyperparameters(
        cls, names: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """Get hyperparameters default values.

        Args:
            names: names of the hyperparameters to choose.
                By default, all available hyperparameters will be suggested.

        Returns:
            a mapping between hyperparameter's name_in_kwargs and its default value (None if not specified)

        """
        if names is None:
            names = cls.get_hyperparameters_names()
        hyperparameters_by_names = cls.get_hyperparameters_by_name()
        return {
            hyperparameters_by_names[name]
            .name_in_kwargs: hyperparameters_by_names[name]
            .default
            for name in names
        }

    @classmethod
    def complete_with_default_hyperparameters(
        cls, kwargs: dict[str, Any], names: Optional[list[str]] = None
    ):
        """Add missing hyperparameters to kwargs by using default values

        Args:
            kwargs: keyword arguments to complete (e.g. for `__init__`, `init_model`, or `solve`)
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
        names: Optional[list[str]] = None,
        kwargs_by_name: Optional[dict[str, dict[str, Any]]] = None,
        fixed_hyperparameters: Optional[dict[str, Any]] = None,
        prefix: str = "",
    ) -> dict[str, Any]:
        """Suggest hyperparameters values during an Optuna trial.

        Args:
            trial: optuna trial during hyperparameters tuning
            names: names of the hyperparameters to choose.
                By default, all available hyperparameters will be suggested.
                If `fixed_hyperparameters` is provided, the corresponding names are removed from `names`.
            kwargs_by_name: options for optuna hyperparameter suggestions, by hyperparameter name
            fixed_hyperparameters: values of fixed hyperparameters, useful for suggesting subbrick hyperparameters,
                if the subbrick class is not suggested by this method, but already fixed.
                Will be added to the suggested hyperparameters.
            prefix: prefix to add to optuna corresponding parameters
              (useful for disambiguating hyperparameters from subsolvers in case of meta-solvers)


        Returns:
            mapping between the hyperparameter name and its suggested value.
            If the hyperparameter has an attribute `name_in_kwargs`, this is used as the key in the mapping
            instead of the actual hyperparameter name.
            the mapping is updated with `fixed_hyperparameters`.

        kwargs_by_name[some_name] will be passed as **kwargs to suggest_hyperparameter_with_optuna(name=some_name)

        """
        if names is None:
            names = cls.get_hyperparameters_names()
        if kwargs_by_name is None:
            kwargs_by_name = {}
        if fixed_hyperparameters is None:
            fixed_hyperparameters = {}
        # Remove fixed hyperparameters from names of hyperparameters to suggest
        names = [name for name in names if name not in fixed_hyperparameters]

        # We suggest the hyperparameters by batch so that hyperparameters depending on other
        # are suggested in a batch after batches involving their depending hyperparameters.
        # Dependency can come from `depends_on` attribute or, in case of `SubBrickKwargsHyperparameter`,
        # from `subbrick_hyperparameter` attribute.
        name2hyperparameter = cls.get_hyperparameters_by_name()
        suggested_hyperparameters = {}
        suggested_and_fixed_hyperparameters = ChainMap(
            suggested_hyperparameters, fixed_hyperparameters
        )
        skipped_hyperparameters: set[str] = set()

        for name in cls._sort_hyperparameters_by_dependency():
            hyperparameter = name2hyperparameter[name]
            kwargs_for_optuna_suggestion = kwargs_by_name.get(name, {})
            kwargs_for_optuna_suggestion["prefix"] = prefix

            # check if dependency condition is fulfilled or not
            if hyperparameter.depends_on is not None:
                previous_hp_name, possible_values = hyperparameter.depends_on
                if previous_hp_name in skipped_hyperparameters or (
                    previous_hp_name in suggested_and_fixed_hyperparameters
                    and suggested_and_fixed_hyperparameters[previous_hp_name]
                    not in possible_values
                ):
                    # condition unfulfilled: do not suggest hyperparameter
                    skipped_hyperparameters.add(name)
                    continue
                elif previous_hp_name not in suggested_and_fixed_hyperparameters:
                    raise ValueError(
                        f"'{hyperparameter.name}' depends on '{previous_hp_name}', "
                        "but the latter has not been suggested by this method "
                        "with `names` including it or being None, nor has it been "
                        "provided via `fixed_hyperparameters`."
                    )

            # subbrickkwargs: add subbrick choice
            if isinstance(hyperparameter, SubBrickKwargsHyperparameter):
                if hyperparameter.subbrick_hyperparameter is None:
                    kwargs_for_optuna_suggestion[
                        "subbrick"
                    ] = hyperparameter.subbrick_cls
                elif (
                    hyperparameter.subbrick_hyperparameter
                    in suggested_and_fixed_hyperparameters
                ):
                    kwargs_for_optuna_suggestion[
                        "subbrick"
                    ] = suggested_and_fixed_hyperparameters[
                        hyperparameter.subbrick_hyperparameter
                    ]
                elif hyperparameter.subbrick_hyperparameter in skipped_hyperparameters:
                    # subbrick_kwargs must be skipped if subbrick itself is skipped
                    skipped_hyperparameters.add(name)
                    continue
                else:
                    raise ValueError(
                        f"'{hyperparameter.name}' needs the choice of '{hyperparameter.subbrick_hyperparameter}', "
                        "but the latter has not been suggested by this method "
                        "with `names` including it or being None, nor has it been "
                        "provided via `fixed_hyperparameters`."
                    )

            # suggest the hyperparameter with optuna if needed
            # NB: we filter the name only now in order to have the skip decision taken before
            # as it could have consequences on hyperparameters further in the dependency graph
            if name in names:
                if hyperparameter.name_in_kwargs is None:
                    key = name
                else:
                    key = hyperparameter.name_in_kwargs
                suggested_and_fixed_hyperparameters[
                    key
                ] = cls.suggest_hyperparameter_with_optuna(
                    trial=trial, name=name, **kwargs_for_optuna_suggestion
                )

        return dict(suggested_and_fixed_hyperparameters)

    @classmethod
    def _get_hyperparameters_dependency_graph(cls) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_nodes_from(cls.get_hyperparameters_names())
        for hp in cls.hyperparameters:
            if hp.depends_on is not None:
                previous_hp_name, _ = hp.depends_on
                g.add_edge(previous_hp_name, hp.name)
            if (
                isinstance(hp, SubBrickKwargsHyperparameter)
                and hp.subbrick_hyperparameter is not None
            ):
                g.add_edge(hp.subbrick_hyperparameter, hp.name)
        return g

    @classmethod
    def _sort_hyperparameters_by_dependency(cls) -> Iterator[str]:
        return nx.algorithms.dag.topological_sort(
            cls._get_hyperparameters_dependency_graph()
        )
