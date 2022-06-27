import os
from datetime import timedelta
from typing import List, Tuple

from discrete_optimization.generic_tools.cp_tools import (
    CPSolverName,
    ParametersCP,
    map_cp_solver_name,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.do_solver import ResultStorage, SolverDO
from discrete_optimization.tsp.common_tools_tsp import build_matrice_distance
from discrete_optimization.tsp.tsp_model import SolutionTSP, TSPModel
from minizinc import Instance, Model, Result, Solver, Status

this_path = os.path.dirname(os.path.abspath(__file__))


class TSP_CPModel:
    FLOAT_VERSION = 0
    INT_VERSION = 1


class TSP_CP_Solver(SolverDO):
    def __init__(
        self,
        tsp_model: TSPModel,
        model_type: TSP_CPModel,
        cp_solver_name: CPSolverName = CPSolverName.CHUFFED,
        params_objective_function: ParamsObjectiveFunction = None,
    ):
        self.tsp_model = tsp_model
        self.model_type = model_type
        self.start_index = self.tsp_model.start_index
        self.end_index = self.tsp_model.end_index
        self.instance = None
        self.cp_solver_name = cp_solver_name
        self.key_decision_variable = ["x"]

        self.distance_matrix = build_matrice_distance(
            self.tsp_model.node_count,
            self.tsp_model.list_points,
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

    def init_model(self, **args):
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

    def solve(self, parameters_cp: ParametersCP = None, **args):
        if parameters_cp is None:
            parameters_cp = ParametersCP.default()
        max_time_seconds = parameters_cp.TimeLimit
        result = self.instance.solve(timeout=timedelta(seconds=max_time_seconds))
        print("Result = ", result)
        circuit = result["x"]
        path = []
        cur_pos = self.start_index
        init = False
        while cur_pos != self.end_index or not init:
            next_pos = circuit[cur_pos] - 1
            path += [next_pos]
            cur_pos = next_pos
            init = True
        var_tsp = SolutionTSP(
            problem=self.tsp_model,
            start_index=self.start_index,
            end_index=self.end_index,
            permutation=path[:-1],
            length=None,
            lengths=None,
        )
        fit = self.aggreg_sol(var_tsp)
        return ResultStorage(
            list_solution_fits=[(var_tsp, fit)],
            mode_optim=self.params_objective_function.sense_function,
        )
