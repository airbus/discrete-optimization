#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Backlog constraints and costs for MILP generic lot sizing solver."""

from discrete_optimization.lotsizing.generic_solver.milp.lotsing_solver_milp import (
    Item,
    LotSizingMilpSolver,
)


class BacklogConstraintMilp(LotSizingMilpSolver[Item]):
    """Backlog constraints and costs for MILP lot sizing solver.

    This mixin adds:
    - Backlog tracking constraints
    - Backlog cost objective terms

    Backlog represents delayed/unmet demand. If backlog is not allowed,
    all backlog variables are forced to zero.
    """

    def create_constraint_backlog(self):
        """Create backlog constraints.

        If backlog is not allowed:
            B_it = 0 for all i, t

        If backlog is allowed:
            B_it = B_i,t-1 + d_it - D_it

        Where:
        - B_it: backlog at end of period t
        - d_it: demand in period t
        - D_it: delivery in period t
        """
        if not self.problem.is_backlog_allowed():
            # Force all backlog to zero
            for item in self.problem.items_list:
                for t in range(self.problem.horizon):
                    self.add_linear_constraint(
                        self.get_backlog_var(item=item, period=t) == 0,
                        name=f"no_backlog_{item}_{t}",
                    )
        else:
            # Backlog accumulation formula
            for item in self.problem.items_list:
                for t in range(self.problem.horizon):
                    demand = int(self.problem.get_demand(item=item, period=t))

                    if t == 0:
                        # B_i0 = d_i0 - D_i0
                        self.add_linear_constraint(
                            self.get_backlog_var(item=item, period=t)
                            == demand - self.get_delivery_var(item=item, period=t),
                            name=f"backlog_balance_{item}_{t}",
                        )
                    else:
                        # B_it = B_i,t-1 + d_it - D_it
                        self.add_linear_constraint(
                            self.get_backlog_var(item=item, period=t)
                            == self.get_backlog_var(item=item, period=t - 1)
                            + demand
                            - self.get_delivery_var(item=item, period=t),
                            name=f"backlog_balance_{item}_{t}",
                        )

    def create_backlog_cost(self):
        """Create backlog penalty cost expression.

        Returns a linear expression for the total backlog cost:
            sum_{i,t} (b_it * B_it)

        Where b_it is the backlog cost per unit per period.

        Returns:
            Linear expression for backlog cost (0 if backlog not allowed)
        """
        if not self.problem.is_backlog_allowed():
            return 0

        backlog_cost_terms = []
        for item in self.problem.items_list:
            for t in range(self.problem.horizon):
                cost_per_unit = int(
                    self.problem.get_backlog_cost_per_unit(item=item, period=t)
                )
                if cost_per_unit > 0:
                    backlog_cost_terms.append(
                        cost_per_unit * self.get_backlog_var(item=item, period=t)
                    )

        if backlog_cost_terms:
            return self.construct_linear_sum(backlog_cost_terms)
        else:
            return 0
