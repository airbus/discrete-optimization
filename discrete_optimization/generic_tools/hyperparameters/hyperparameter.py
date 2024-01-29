#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional


@dataclass
class Hyperparameter:
    name: str
    default: Optional[Any] = None


@dataclass
class IntegerHyperparameter(Hyperparameter):
    low: Optional[int] = None
    high: Optional[int] = None
    default: Optional[int] = None


@dataclass
class FloatHyperparameter(Hyperparameter):
    low: Optional[float] = None
    high: Optional[float] = None
    default: Optional[float] = None


@dataclass
class CategoricalHyperparameter(Hyperparameter):
    choices: List[Any] = field(default_factory=list)


class EnumHyperparameter(CategoricalHyperparameter):
    def __init__(self, name: str, enum: Enum, default: Optional[Any] = None):
        super().__init__(name, choices=list(enum), default=default)
        self.enum = enum
