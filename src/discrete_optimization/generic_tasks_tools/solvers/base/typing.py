#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any

Scalar = bool | int

LinearExpr = Any
"""Linear expression.

According to framework, will correspond to
- ortools/cpsat: LinearExprT
- optalcp: int | bool | IntExpr
- cpmpy: int | bool | Expression

"""

IntervalVar = Any
"""Interval variable.

Will correspond to the porper `IntervalVar` according to the framework.

"""

Model = Any
"""Cp model."""


Constraint = Any

BoolExpr = Any

CpVar = Any
"""Variable from the cp model."""
