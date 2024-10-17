#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any, Optional

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_problem import Problem, Solution
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.gpdp.problem import ProxyClass
from discrete_optimization.gpdp.solvers.ortools_routing import (
    GpdpSolution,
    OrtoolsGpdpSolver,
)
from discrete_optimization.tsp.problem import TspSolution
from discrete_optimization.tsp.solvers import TspSolver


class GpdpBasedTspSolver(TspSolver, WarmstartMixin):
    def __init__(self, problem: Problem, **kwargs: Any):
        super().__init__(problem, **kwargs)
        self.gpdp_problem = ProxyClass.from_tsp_to_gpdp(tsp_model=self.problem)
        self.solver: OrtoolsGpdpSolver = None

    def init_model(self, **kwargs: Any) -> None:
        solver = OrtoolsGpdpSolver(self.gpdp_problem)
        solver.init_model(time_limit=kwargs["time_limit"], include_time_dimension=False)
        self.solver = solver

    def set_warm_start(self, solution: Solution) -> None:
        solution: TspSolution
        sol = GpdpSolution(
            problem=self.gpdp_problem,
            trajectories={
                0: [self.gpdp_problem.origin_vehicle[0]]
                + solution.permutation_from0
                + [self.gpdp_problem.target_vehicle[0]]
            },
            times=None,
            resource_evolution=None,
        )
        self.solver.set_warm_start(sol)

    def solve(
        self,
        callbacks: Optional[list[Callback]] = None,
        time_limit: Optional[float] = 100.0,
        **kwargs: Any
    ) -> ResultStorage:
        if self.solver is None:
            self.init_model(time_limit=time_limit, **kwargs)
        res = self.solver.solve(callbacks=callbacks, **kwargs)
        res_ = self.create_result_storage([])
        for sol, _ in res.list_solution_fits:
            sol: GpdpSolution
            sol_tsp = TspSolution(
                problem=self.problem,
                start_index=self.problem.start_index,
                end_index=self.problem.end_index,
                permutation_from0=sol.trajectories[0][1:-1],
            )
            res_.append((sol_tsp, self.aggreg_from_sol(sol_tsp)))
        return res_
