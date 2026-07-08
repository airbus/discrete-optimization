#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Parallel production constraints for MILP generic lot sizing solver."""

from discrete_optimization.lotsizing.generic_solver.milp.lotsing_solver_milp import (
    Item,
    LotSizingMilpSolver,
)


class ParallelProductionConstraintMilp(LotSizingMilpSolver[Item]):
    """Parallel production constraints for MILP lot sizing solver.

    This mixin adds constraints to enforce exclusive production when
    parallel production is not allowed (only one item can be produced
    per period).
    """

    def create_constraint_parallel_production(self):
        """Create parallel production constraints.

        If parallel production is not allowed, enforce that at most one
        item can be produced in each period:
            sum_i Y_it <= 1 for all t

        Where Y_it is the setup binary variable (1 if item i is produced in period t).
        """
        if not self.problem.allows_parallel_production():
            for t in range(self.problem.horizon):
                setup_vars = [
                    self.get_production_binary_var(item=item, period=t)
                    for item in self.problem.items_list
                ]
                self.add_linear_constraint(
                    self.construct_linear_sum(setup_vars) <= 1,
                    name=f"at_most_one_item_t{t}",
                )
