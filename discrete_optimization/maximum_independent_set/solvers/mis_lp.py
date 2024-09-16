from collections.abc import Callable
from typing import Any

from discrete_optimization.generic_tools.lp_tools import MilpSolver
from discrete_optimization.maximum_independent_set.mis_model import MisSolution
from discrete_optimization.maximum_independent_set.solvers.mis_solver import MisSolver


class BaseLPMisSolver(MisSolver, MilpSolver):
    vars_node: dict[int, Any]

    def retrieve_current_solution(
        self,
        get_var_value_for_current_solution: Callable[[Any], float],
        get_obj_value_for_current_solution: Callable[[], float],
    ) -> MisSolution:

        chosen = [0] * self.problem.number_nodes

        for i in range(0, self.problem.number_nodes):
            if get_var_value_for_current_solution(self.vars_node[i]) > 0.5:
                chosen[i] = 1

        return MisSolution(self.problem, chosen)

    def convert_to_variable_values(self, solution: MisSolution) -> dict[Any, float]:
        """Convert a solution to a mapping between model variables and their values.

        Will be used by set_warm_start().

        """
        return {
            self.vars_node[i]: solution.chosen[i]
            for i in range(0, self.problem.number_nodes)
        }
