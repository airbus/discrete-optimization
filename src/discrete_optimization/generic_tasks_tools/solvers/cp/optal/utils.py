#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

from discrete_optimization.generic_tasks_tools.enums import StartOrEnd
from discrete_optimization.generic_tasks_tools.solvers.cp.base.typing import LinearExpr

try:
    import optalcp as cp
except ImportError:
    cp = None


def get_interval_start_or_end_variable(
    itv: cp.IntervalVar, start_or_end: StartOrEnd
) -> LinearExpr:
    """Retrieve the variable storing the start or end time of given interval.

    Args:
        itv:
        start_or_end:

    Returns:

    """
    if start_or_end == StartOrEnd.START:
        return itv.start()
    else:
        return itv.end()
