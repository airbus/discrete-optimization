from __future__ import annotations

import json
from collections.abc import Hashable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

import pandas as pd

from discrete_optimization.generic_tools.do_solver import SolverDO, StatusSolver

# keys
ID = "id"
SOLVER = "solver"
PARAMETERS = "parameters"
NAME = "name"
STATUS = "status"
INSTANCE = "instance"
METRICS = "metrics"
CONFIG_PREFIX = "config_"
REASON = "reason"
CONFIG = "config"

# Type alias
ConfigDict = dict[str, Union["ConfigDict", Hashable]]


@dataclass
class SolverConfig:
    cls: type[SolverDO]
    kwargs: dict[str, Any]


@dataclass
class SolverJsonableConfig:
    """Config representing solver used with its tuning."""

    solver: str
    parameters: ConfigDict
    name: Optional[str] = None

    def get_json_parameters(self) -> str:
        """Jsonify the config parameters attribute."""
        return json.dumps(self.parameters, cls=DoJSONEncoder)

    def get_record(self) -> dict[str, Optional[str]]:
        return {
            SOLVER: self.solver,
            PARAMETERS: self.get_json_parameters(),
            NAME: self.name,
        }

    def as_nested_dict(self) -> ConfigDict:
        return {
            SOLVER: self.solver,
            PARAMETERS: self.parameters,
            NAME: self.name,
        }

    @classmethod
    def from_xp_metadata_record(cls, record):
        return cls(
            solver=record[CONFIG_PREFIX + SOLVER],
            parameters=json.loads(record[CONFIG_PREFIX + PARAMETERS]),
            name=record[CONFIG_PREFIX + NAME],
        )

    @classmethod
    def from_record(cls, record):
        return cls(
            solver=record[SOLVER],
            parameters=json.loads(record[PARAMETERS]),
            name=record[NAME],
        )

    @classmethod
    def from_solver_config(cls, config: SolverConfig, name: Optional[str] = None):
        return cls(name=name, solver=config.cls.__name__, parameters=config.kwargs)


@dataclass
class Experiment:
    """Experiment of a d-o study."""

    xp_id: int
    instance: str
    status: str
    config: SolverJsonableConfig
    metrics: pd.DataFrame  # time-indexed dataframe, each column being a tracked metric
    reason: str = ""

    def get_metadata_as_a_record(self) -> dict[str, Any]:
        return {
            ID: self.xp_id,
            INSTANCE: self.instance,
            **{CONFIG_PREFIX + k: v for k, v in self.config.get_record().items()},
            STATUS: self.status,
            REASON: self.reason,
        }

    def get_metadata_as_nested_dict(self) -> ConfigDict:
        return {
            ID: self.xp_id,
            INSTANCE: self.instance,
            CONFIG: self.config.as_nested_dict(),
            STATUS: self.status,
            REASON: self.reason,
        }

    def to_df(self):
        """Convert to a dataframe (metrics) with metadata store in `attrs` attribute."""
        metadata = self.get_metadata_as_nested_dict()
        df = pd.DataFrame(self.metrics)
        df.attrs = metadata
        return df

    @classmethod
    def from_solver_config(
        cls,
        xp_id: int,
        instance: str,
        status: Union[str, StatusSolver],
        solver_config: SolverConfig,
        metrics: pd.DataFrame,
        config_name: Optional[str] = None,
        reason: str = "",
    ):
        if isinstance(status, StatusSolver):
            status_str = status.value
        else:
            status_str = status
        return cls(
            xp_id=xp_id,
            instance=instance,
            status=status_str,
            config=SolverJsonableConfig.from_solver_config(
                solver_config, name=config_name
            ),
            metrics=metrics,
            reason=reason,
        )


@dataclass
class Study:
    name: str
    instances: list[str]
    solver_configs: list[SolverConfig]


class DoJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if hasattr(o, "to_json"):
            return o.to_json()
        elif isinstance(o, Enum):
            return o.value
        elif hasattr(o, "__dict__"):
            return o.__dict__
        return super().default(o)
