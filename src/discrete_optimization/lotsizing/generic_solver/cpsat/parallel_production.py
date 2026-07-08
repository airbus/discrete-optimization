#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.lotsizing.generic_solver.cpsat.lotsing_solver_cpsat import (
    Item,
    LotSizingCpSatSolver,
)


class ParallelProductionConstraintCpsat(LotSizingCpSatSolver[Item]):
    def create_constraint_parallel_production(self):
        if not self.problem.allows_parallel_production():
            for t in range(self.problem.horizon):
                self.cp_model.add_at_most_one(
                    [
                        self.get_production_binary_var(item=item, period=t)
                        for item in self.problem.items_list
                    ]
                )
