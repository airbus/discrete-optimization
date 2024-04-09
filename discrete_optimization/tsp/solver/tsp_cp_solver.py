#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Any, List, Optional

from minizinc import Instance, Model, Solver

from discrete_optimization.generic_tools.cp_tools import (
    CPSolverName,
    MinizincCPSolver,
    find_right_minizinc_solver_name,
)
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.tsp.common_tools_tsp import build_matrice_distance
from discrete_optimization.tsp.solver.tsp_solver import SolverTSP
from discrete_optimization.tsp.tsp_model import SolutionTSP, TSPModel

logger = logging.getLogger(__name__)
this_path = os.path.dirname(os.path.abspath(__file__))


class TSP_CPModel:
    FLOAT_VERSION = 0
    INT_VERSION = 1


class TSP_CP_Solver(MinizincCPSolver, SolverTSP):
    def __init__(
        self,
        problem: TSPModel,
        model_type: TSP_CPModel,
        cp_solver_name: CPSolverName = CPSolverName.CHUFFED,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        silent_solve_error: bool = False,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.silent_solve_error = silent_solve_error
        self.model_type = model_type
        self.start_index = self.problem.start_index
        self.end_index = self.problem.end_index
        self.cp_solver_name = cp_solver_name
        self.key_decision_variable = ["x"]

        self.distance_matrix = build_matrice_distance(
            self.problem.node_count,
            method=self.problem.evaluate_function_indexes,
        )
        self.distance_matrix[self.end_index, self.start_index] = 0
        self.distance_list_2d = [
            [
                int(x) if model_type == TSP_CPModel.INT_VERSION else x
                for x in self.distance_matrix[i, :]
            ]
            for i in range(self.distance_matrix.shape[0])
        ]

    def init_model(self, **args: Any) -> None:
        if self.model_type == TSP_CPModel.FLOAT_VERSION:
            model = Model(os.path.join(this_path, "../minizinc/tsp_float.mzn"))
        if self.model_type == TSP_CPModel.INT_VERSION:
            model = Model(os.path.join(this_path, "../minizinc/tsp_int.mzn"))
        # Find the MiniZinc solver configuration for Gecode
        solver = Solver.lookup(find_right_minizinc_solver_name(self.cp_solver_name))
        # Create an Instance of the n-Queens model for Gecode
        instance = Instance(solver, model)
        instance["n"] = self.problem.node_count
        instance["distances"] = self.distance_list_2d
        instance["start"] = self.start_index + 1
        instance["end"] = self.end_index + 1
        self.instance = instance

    def retrieve_solution(
        self, _output_item: Optional[str] = None, **kwargs: Any
    ) -> SolutionTSP:
        """Return a d-o solution from the variables computed by minizinc.

        Args:
            _output_item: string representing the minizinc solver output passed by minizinc to the solution constructor
            **kwargs: keyword arguments passed by minzinc to the solution contructor
                containing the objective value (key "objective"),
                and the computed variables as defined in minizinc model.

        Returns:

        """
        circuit = kwargs["x"]
        return self._retrieve_solution_from_circuit(circuit)

    def _retrieve_solution_from_circuit(self, circuit: List[int]) -> SolutionTSP:
        path = []
        cur_pos = self.start_index
        init = False
        while cur_pos != self.end_index or not init:
            next_pos = circuit[cur_pos] - 1
            path += [next_pos]
            cur_pos = next_pos
            init = True
        return SolutionTSP(
            problem=self.problem,
            start_index=self.start_index,
            end_index=self.end_index,
            permutation=path[:-1],
            length=None,
            lengths=None,
        )
