#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.lotsizing.generic_solver.cpsat.lotsing_solver_cpsat import (
    Item,
    LotSizingCpSatSolver,
)


class BacklogGenericSolver(LotSizingCpSatSolver[Item]):
    def create_constraint_backlog(self):
        if not self.problem.is_backlog_allowed():
            for item in self.problem.items_list:
                for t in self.problem.horizon:
                    self.cp_model.add(self.get_backlog_var(item=item, period=t) == 0)
