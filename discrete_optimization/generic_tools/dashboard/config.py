from __future__ import annotations

from collections.abc import Hashable
from typing import Any, Optional, Union

from discrete_optimization.generic_tools.study.experiment import NAME, SOLVER

# type aliases
DictConfig = Union[dict[str, "DictConfig"], Hashable]
HashableConfig = Union[tuple[str, "HashableConfig"], Hashable]


def get_config_name(config: DictConfig) -> str:
    if isinstance(config, dict):
        if NAME in config:
            return config[NAME]
        elif SOLVER in config:
            return f"{config[SOLVER]}-{get_config_name({k: v for k,v in config.items() if k != SOLVER})}"
        else:
            return "-".join(f"{k}-{get_config_name(v)}" for k, v in config.items())
    else:
        return str(config)


def convert_config_dict2hashable(config: DictConfig) -> HashableConfig:
    if isinstance(config, dict):
        return tuple((k, convert_config_dict2hashable(v)) for k, v in config.items())
    else:
        return config


def is_tupleddict(config: Any) -> bool:
    return isinstance(config, tuple) and all(
        isinstance(configitem, tuple)
        and len(configitem) == 2
        and isinstance(configitem[0], str)
        for configitem in config
    )


def convert_config_hashable2dict(config: HashableConfig) -> DictConfig:
    if is_tupleddict(config):
        return {k: convert_config_hashable2dict(v) for k, v in config}
    else:
        return config


class ConfigStore:
    """Store experiments config and mapping to their names"""

    def __init__(self):
        self.map_name2config: dict[str, HashableConfig] = {}
        self.map_config2name: dict[HashableConfig, str] = {}
        self.map_config2hasusername: dict[HashableConfig, bool] = {}

    def add(self, config: DictConfig) -> None:
        """Add a config to the store.

        Ensure bijection between names and configs.
        If name already given, use it.
        If not, construct it from solver parameters.
        If 2 names given in different occurences raise an error.
        If 2 different config share the same name, raise an error.

        """
        config = dict(config)  # copy dict to avoid inplace modification
        name: Optional[str] = None
        # name given by user?
        if NAME in config:
            name = config[NAME]
            del config[NAME]
        # hashable version of the config
        hashable_config = convert_config_dict2hashable(config)

        # get a name and check that previous similar config share the name (or had not already one defined)
        if (
            hashable_config in self.map_config2name
        ):  # config already added (with potentially no name or another one)
            if name is None:
                name = self.map_config2name[hashable_config]
            else:
                if self.map_config2hasusername[
                    hashable_config
                ]:  # had already a name given by user
                    assert name == self.map_config2name[hashable_config], (
                        "Same configs should share same names. "
                        f"The names {name} and {self.map_config2name[hashable_config]} were used."
                    )
                else:  # stored name was a name constructed by default => replace it with name given by user
                    self.map_config2name[hashable_config] = name
                    self.map_config2hasusername[hashable_config] = True
        else:
            if name is None:  # no name given by user: construct one from parameters
                name = get_config_name(config)
                self.map_config2hasusername[hashable_config] = False
            else:  # name given by user
                self.map_config2hasusername[hashable_config] = True
            self.map_config2name[hashable_config] = name

        # check that only one config corresponds to this name
        if name in self.map_name2config:
            assert hashable_config == self.map_name2config[name], (
                f"Two configs share same name {name}: "
                f"{convert_config_hashable2dict(self.map_name2config[name])} and {config}."
            )
        else:
            self.map_name2config[name] = hashable_config

    def get_name(self, config: DictConfig) -> str:
        config = dict(config)  # copy dict to avoid inplace modification
        if NAME in config:
            del config[NAME]
        hashable_config = convert_config_dict2hashable(config)
        if hashable_config not in self.map_config2name:
            raise RuntimeError(
                "Be sure to add all configs to the store before extracting names. "
                "That way, we can ensure same configs share names "
                "and that no name is used for several different configs."
            )
        else:
            return self.map_config2name[hashable_config]

    def get_config(self, name: str) -> DictConfig:
        return convert_config_hashable2dict(self.map_name2config[name])
