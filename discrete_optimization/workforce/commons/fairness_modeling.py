from dataclasses import dataclass
from enum import Enum


class ModelisationDispersion(Enum):
    EXACT_MODELING_WITH_IMPLICATION = 0
    EXACT_MODELING_DUPLICATED_VARS = 1
    MAX_DIFF = 2
    PROXY_MAX_MIN = 3
    PROXY_MIN_MAX = 4
    PROXY_SUM = 5
    PROXY_SLACK = 6
    PROXY_GINI_INDICATOR = 7
