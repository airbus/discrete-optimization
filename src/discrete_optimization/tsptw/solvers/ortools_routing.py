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
from discrete_optimization.tsptw.problem import TSPTWProblem, TSPTWSolution


class OrtoolsTspTwSolver(SolverDO):
    problem: TSPTWProblem
    gpdp_problem: GpdpProblem
    solver: OrtoolsGpdpSolver

    def init_model(self, time_limit=10, scaling: float = 100, **kwargs: Any) -> None:
        gpdp_problem = ProxyClass.from_tsptw_to_gpdp(self.problem, True)
        solver = OrtoolsGpdpSolver(problem=gpdp_problem, factor_multiplier_time=scaling)
        solver.init_model(
            one_visit_per_node=True,
            include_time_windows=True,
            include_time_dimension=True,
            local_search_metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
            first_solution_strategy=routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
            time_limit=time_limit,
            parameters_cost=[ParametersCost(dimension_name="Time", global_span=True)],
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
        tsp_tw_sol = TSPTWSolution(
            problem=self.problem, permutation=[i + 1 for i in sol.trajectories[0][1:-1]]
        )
        fit = self.aggreg_from_sol(tsp_tw_sol)
        result_storage.append((tsp_tw_sol, fit))
        callback.on_solve_end(result_storage, self)
        return result_storage
