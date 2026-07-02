#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Instance generator for capacitated lot sizing with setup times problems."""

from __future__ import annotations

import numpy as np

from discrete_optimization.lotsizing.capacitatedsetuptimes.problem import (
    CapacitatedSetupTimesLSP,
)


def create_simple_instance(
    nb_items: int = 3,
    horizon: int = 6,
    capacity: float = 10.0,
    setup_time: float = 2.0,
    allow_delays: bool = True,
) -> CapacitatedSetupTimesLSP:
    """Create a simple test instance with setup times.

    Args:
        nb_items: Number of items
        horizon: Number of periods
        capacity: Production capacity per period
        setup_time: Fixed setup time for all items/periods
        allow_delays: Whether backlog/delays are allowed (default: True for feasibility)

    Returns:
        CapacitatedSetupTimesLSP instance
    """
    # Simple demands pattern
    demands = np.random.randint(1, 5, size=(nb_items, horizon))

    # Constant setup times
    setup_times = np.full((nb_items, horizon), setup_time, dtype=np.float64)

    # Symmetric changeover costs (distance between items)
    changeover_costs = np.zeros((nb_items, nb_items), dtype=np.int64)
    for i in range(nb_items):
        for j in range(nb_items):
            changeover_costs[i, j] = abs(i - j) * 10

    # Simple stock costs
    stock_costs = np.ones(nb_items, dtype=np.float64)

    return CapacitatedSetupTimesLSP(
        nb_items=nb_items,
        horizon=horizon,
        demands=demands,
        capacity_machine=capacity,
        setup_times=setup_times,
        changeover_costs=changeover_costs,
        stock_cost_per_type=stock_costs,
        allow_delays=allow_delays,
    )
