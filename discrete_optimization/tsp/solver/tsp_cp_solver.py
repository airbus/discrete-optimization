#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Any, List, Optional, Tuple

from minizinc import Instance, Model, Result, Solver

from discrete_optimization.generic_tools.cp_tools import (
    CPSolverName,
    MinizincCPSolver,
    ParametersCP,
    map_cp_solver_name,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
    fitness_class,
)
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
        tsp_model: TSPModel,
        model_type: TSP_CPModel,
        cp_solver_name: CPSolverName = CPSolverName.CHUFFED,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        silent_solve_error: bool = False,
    ):
        SolverTSP.__init__(self, tsp_model=tsp_model)
        self.silent_solve_error = silent_solve_error
        self.model_type = model_type
        self.start_index = self.tsp_model.start_index
        self.end_index = self.tsp_model.end_index
        self.cp_solver_name = cp_solver_name
        self.key_decision_variable = ["x"]

        self.distance_matrix = build_matrice_distance(
            self.tsp_model.node_count,
            method=self.tsp_model.evaluate_function_indexes,
        )
        self.distance_matrix[self.end_index, self.start_index] = 0
        self.distance_list_2d = [
            [
                int(x) if model_type == TSP_CPModel.INT_VERSION else x
                for x in self.distance_matrix[i, :]
            ]
            for i in range(self.distance_matrix.shape[0])
        ]
        (
            self.aggreg_sol,
            self.aggreg_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.tsp_model, params_objective_function=params_objective_function
        )

    def init_model(self, **args: Any) -> None:
        if self.model_type == TSP_CPModel.FLOAT_VERSION:
            model = Model(os.path.join(this_path, "../minizinc/tsp_float.mzn"))
        if self.model_type == TSP_CPModel.INT_VERSION:
            model = Model(os.path.join(this_path, "../minizinc/tsp_int.mzn"))
        # Find the MiniZinc solver configuration for Gecode
        solver = Solver.lookup(map_cp_solver_name[self.cp_solver_name])
        # Create an Instance of the n-Queens model for Gecode
        instance = Instance(solver, model)
        instance["n"] = self.tsp_model.node_count
        instance["distances"] = self.distance_list_2d
        instance["start"] = self.start_index + 1
        instance["end"] = self.end_index + 1
        self.instance = instance

    def retrieve_solutions(
        self, result: Result, parameters_cp: ParametersCP
    ) -> ResultStorage:
        intermediate_solutions = parameters_cp.intermediate_solution
        solutions_fit: List[Tuple[Solution, fitness_class]] = []
        if intermediate_solutions:
            for i in range(len(result)):
                circuit = result[i, "x"]
                var_tsp = self._retrieve_solution_from_circuit(circuit)
                fit = self.aggreg_sol(var_tsp)
                solutions_fit.append((var_tsp, fit))
        else:
            circuit = result["x"]
            var_tsp = self._retrieve_solution_from_circuit(circuit)
            fit = self.aggreg_sol(var_tsp)
            solutions_fit.append((var_tsp, fit))
        return ResultStorage(
            list_solution_fits=solutions_fit,
            mode_optim=self.params_objective_function.sense_function,
        )

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
            problem=self.tsp_model,
            start_index=self.start_index,
            end_index=self.end_index,
            permutation=path[:-1],
            length=None,
            lengths=None,
        )
