#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Capacitated lot sizing problem with setup times.

CLSP with setup times (from page 11 of slides_313.pdf):
- Setup times τ_it consume capacity when production occurs
- Capacity constraint: Σ_i (p_it·X_it + τ_it·Y_it) ≤ h_t
"""

from discrete_optimization.lotsizing.capacitatedsetuptimes.problem import (
    CapacitatedSetupTimesLSP,
    CapacitatedSetupTimesSolution,
)

__all__ = [
    "CapacitatedSetupTimesLSP",
    "CapacitatedSetupTimesSolution",
]
