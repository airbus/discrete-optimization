#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.pickup_vrp.gpdp import GPDP


class SolverPickupVrp(SolverDO):
    problem: GPDP
