#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations  # see annotations as str

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Optional

if TYPE_CHECKING:  # only for type checkers
    try:
        import optuna
    except ImportError:
        pass


@dataclass
class Hyperparameter:
    name: str
    default: Optional[Any] = None

    def suggest_with_optuna(self, trial: optuna.trial.Trial, **kwargs: Any) -> Any:
        ...


@dataclass
class IntegerHyperparameter(Hyperparameter):
    low: Optional[int] = None
    high: Optional[int] = None
    default: Optional[int] = None

    def suggest_with_optuna(self, trial: optuna.trial.Trial, **kwargs: Any) -> Any:
        if self.low is not None and "low" not in kwargs:
            kwargs["low"] = self.low
        if self.high is not None and "high" not in kwargs:
            kwargs["high"] = self.high
        return trial.suggest_int(name=self.name, **kwargs)


@dataclass
class FloatHyperparameter(Hyperparameter):
    low: Optional[float] = None
    high: Optional[float] = None
    default: Optional[float] = None

    def suggest_with_optuna(self, trial: optuna.trial.Trial, **kwargs: Any) -> Any:
        if self.low is not None and "low" not in kwargs:
            kwargs["low"] = self.low
        if self.high is not None and "high" not in kwargs:
            kwargs["high"] = self.high
        return trial.suggest_float(name=self.name, **kwargs)


@dataclass
class CategoricalHyperparameter(Hyperparameter):
    choices: List[Any] = field(default_factory=list)

    def suggest_with_optuna(self, trial: optuna.trial.Trial, **kwargs: Any) -> Any:
        if self.choices is not None and "choices" not in kwargs:
            kwargs["choices"] = self.choices
        return trial.suggest_categorical(name=self.name, **kwargs)


class EnumHyperparameter(CategoricalHyperparameter):
    def __init__(self, name: str, enum: Enum, default: Optional[Any] = None):
        super().__init__(name, choices=list(enum), default=default)
        self.enum = enum

    def suggest_with_optuna(self, trial: optuna.trial.Trial, **kwargs: Any) -> Any:
        choices = kwargs.get("choices", self.choices)
        choices_str = [c.name for c in choices]
        choice_str = trial.suggest_categorical(name=self.name, choices=choices_str)
        return self.enum[choice_str]
