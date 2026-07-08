#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.lotsizing.generic_solver.cpsat.lotsing_solver_cpsat import (
    Item,
    LotSizingCpSatSolver,
)


class BacklogConstraintCpsat(LotSizingCpSatSolver[Item]):
    def create_constraint_backlog(self):
        if not self.problem.is_backlog_allowed():
            for item in self.problem.items_list:
                for t in range(self.problem.horizon):
                    self.cp_model.add(self.get_backlog_var(item=item, period=t) == 0)
        else:
            # Formula to define backlog.
            for item in self.problem.items_list:
                for t in range(self.problem.horizon):
                    if t == 0:
                        self.cp_model.add(
                            self.get_backlog_var(item=item, period=t)
                            == self.problem.get_demand(item=item, period=t)
                            - self.get_delivery_var(item=item, period=t)
                        )
                    else:
                        self.cp_model.add(
                            self.get_backlog_var(item=item, period=t)
                            == self.get_backlog_var(item=item, period=t - 1)
                            + self.problem.get_demand(item=item, period=t)
                            - self.get_delivery_var(item=item, period=t)
                        )

    def create_backlog_cost(self):
        if not self.problem.is_backlog_allowed():
            return 0
        return sum(
            [
                self.get_backlog_var(item=item, period=t)
                * int(self.problem.get_backlog_cost_per_unit(item=item, period=t))
                for item in self.problem.items_list
                for t in range(self.problem.horizon)
            ]
        )
