import logging
from typing import Any, Optional

import tqdm
from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import CpSolverSolutionCallback

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCPSatSolver
from discrete_optimization.maximum_independent_set.mis_model import (
    MisProblem,
    MisSolution,
)
from discrete_optimization.maximum_independent_set.solvers.mis_solver import MisSolver

logger = logging.getLogger(__file__)


class MisOrtoolsSolver(MisSolver, OrtoolsCPSatSolver, WarmstartMixin):
    def __init__(
        self,
        problem: MisProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = None

    def init_model(self, **kwargs):
        model = cp_model.CpModel()
        in_set = [
            model.NewBoolVar(f"in_set_{i}") for i in range(self.problem.number_nodes)
        ]
        # Add constraints to ensure that adjacent vertices are not both in the independent set.
        for edge in tqdm.tqdm(self.problem.edges):
            model.AddBoolOr(
                [
                    in_set[self.problem.nodes_to_index[edge[0]]].Not(),
                    in_set[self.problem.nodes_to_index[edge[1]]].Not(),
                ]
            )
            model.AddAtMostOne(
                [
                    in_set[self.problem.nodes_to_index[edge[0]]],
                    in_set[self.problem.nodes_to_index[edge[1]]],
                ]
            )

            model.Add(
                in_set[self.problem.nodes_to_index[edge[0]]]
                + in_set[self.problem.nodes_to_index[edge[1]]]
                <= 1
            )
        # Maximize the sum of weights of the independent set.
        if self.problem.attribute_aggregate == "size":
            objective = model.NewIntVar(0, self.problem.number_nodes, "objective")
            model.Add(objective == sum(in_set))
        else:
            objective = model.NewIntVar(
                0, int(sum(self.problem.attr_list)), "objective"
            )
            model.Add(
                objective
                == sum(
                    [
                        in_set[i] * int(self.problem.attr_list[i])
                        for i in range(len(in_set))
                    ]
                )
            )
        model.Maximize(objective)
        self.cp_model = model
        self.variables = {"in_set": in_set, "objective": objective}

    def set_warm_start(self, solution: MisSolution) -> None:
        """Make the solver warm start from the given solution."""
        self.cp_model.clear_hints()
        for i in range(len(solution.chosen)):
            self.cp_model.AddHint(self.variables["in_set"][i], solution.chosen[i])

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> MisSolution:

        chosen = [0] * self.problem.number_nodes

        for i in range(0, self.problem.number_nodes):
            chosen[i] = cpsolvercb.Value(self.variables["in_set"][i])

        return MisSolution(self.problem, chosen)
