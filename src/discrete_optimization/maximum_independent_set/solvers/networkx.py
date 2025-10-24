from typing import Any, Optional

import networkx as nx

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.maximum_independent_set.problem import MisSolution
from discrete_optimization.maximum_independent_set.solvers.mis_solver import MisSolver


class NetworkxMisSolver(MisSolver):
    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        sol = nx.approximation.maximum_independent_set(self.problem.graph_nx)
        chosen = [
            1 if self.problem.nodes[i] in sol else 0
            for i in range(self.problem.number_nodes)
        ]
        solution = MisSolution(problem=self.problem, chosen=chosen)
        fit = self.aggreg_from_sol(solution)
        return self.create_result_storage(
            [(solution, fit)],
        )
