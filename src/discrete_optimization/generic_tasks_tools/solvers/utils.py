#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from typing import Any


def is_a_trivial_zero(var: Any) -> bool:
    """Return whether the (cpsat, optalcp, ...) variable is actually a plain 0 integer.

    For instance, tells if is_present variables are real variables or not to avoid
    including them in sum, max, ...

    """
    return isinstance(var, int) and var == 0
