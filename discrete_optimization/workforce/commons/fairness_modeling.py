#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from enum import Enum


class ModelisationDispersion(Enum):
    EXACT_MODELING_WITH_IMPLICATION = 0
    EXACT_MODELING_DUPLICATED_VARS = 1
    MAX_DIFF = 2
    PROXY_MAX_MIN = 3
    PROXY_MIN_MAX = 4
    PROXY_SUM = 5
    PROXY_SLACK = 6
