#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Any, Optional

from ortools.constraint_solver import routing_enums_pb2

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.gpdp.problem import GpdpProblem, GpdpSolution, ProxyClass
from discrete_optimization.gpdp.solvers import GpdpSolver
from discrete_optimization.gpdp.solvers.ortools_routing import (
    OrtoolsGpdpSolver,
    ParametersCost,
)
from discrete_optimization.vrptw.problem import VRPTWProblem, VRPTWSolution


class OrtoolsVrpTwSolver(SolverDO):
    problem: VRPTWProblem
    gpdp_problem: GpdpProblem
    solver: OrtoolsGpdpSolver

    def init_model(
        self,
        time_limit=10,
        scaling: float = 100,
        cost_per_vehicle: int = 100000,
        **kwargs: Any,
    ) -> None:
        gpdp_problem = ProxyClass.from_vrptw_to_gpdp(self.problem, True)
        solver = OrtoolsGpdpSolver(problem=gpdp_problem, factor_multiplier_time=scaling)
        solver.init_model(
            one_visit_per_node=True,
            include_time_windows=True,
            include_demand=True,
            neg_capacity_version=False,
            include_time_dimension=True,
            consider_empty_route_cost=False,
            cost_per_vehicle_used=cost_per_vehicle,
            local_search_metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
            first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC,
            time_limit=time_limit,
            parameters_cost=[
                ParametersCost(
                    dimension_name="Time",
                    global_span=False,
                    coefficient_vehicles=[1] * gpdp_problem.number_vehicle,
                )
            ],
        )
        self.solver = solver

    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        callback = CallbackList(callbacks)
        callback.on_solve_start(self)
        result_storage = self.create_result_storage([])
        res = self.solver.solve()
        sol: GpdpSolution = res[-1][0]
        vrp_tw_sol = VRPTWSolution(
            problem=self.problem,
            routes=[
                [x + 1 for x in sol.trajectories[i][1:-1]]
                for i in range(len(sol.trajectories))
            ],
        )
        vrp_tw_sol.routes = [r for r in vrp_tw_sol.routes if len(r) > 1]
        print(vrp_tw_sol.routes)
        fit = self.aggreg_from_sol(vrp_tw_sol)
        result_storage.append((vrp_tw_sol, fit))
        callback.on_solve_end(result_storage, self)
        return result_storage
