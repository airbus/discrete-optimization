#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from ortools.sat.python.cp_model import CpModel, CpSolverSolutionCallback, IntVar

from discrete_optimization.coloring.coloring_model import ColoringSolution
from discrete_optimization.coloring.solvers.coloring_solver_with_starting_solution import (
    SolverColoringWithStartingSolution,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Problem,
    Solution,
)
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    EnumHyperparameter,
)
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCPSatSolver


class ModelingCPSat(Enum):
    BINARY = 0
    INTEGER = 1


class ColoringCPSatSolver(
    OrtoolsCPSatSolver, SolverColoringWithStartingSolution, WarmstartMixin
):
    hyperparameters = [
        EnumHyperparameter(
            name="modeling", enum=ModelingCPSat, default=ModelingCPSat.INTEGER
        ),
        CategoricalHyperparameter(
            name="warmstart", choices=[True, False], default=True
        ),
        CategoricalHyperparameter(
            name="value_sequence_chain",
            choices=[True, False],
            default=False,
            depends_on=("modeling", [ModelingCPSat.INTEGER]),
        ),
        CategoricalHyperparameter(
            name="used_variable",
            choices=[True, False],
            default=False,
            depends_on=("modeling", [ModelingCPSat.INTEGER]),
        ),
        CategoricalHyperparameter(
            name="symmetry_on_used",
            choices=[True, False],
            default=True,
            depends_on=("modeling", [ModelingCPSat.INTEGER]),
        ),
    ] + SolverColoringWithStartingSolution.hyperparameters

    def __init__(
        self,
        problem: Problem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.modeling: Optional[ModelingCPSat] = None
        self.variables: Dict[str, Union[List[IntVar], List[Dict[int, IntVar]]]] = {}

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> Solution:
        if self.modeling == ModelingCPSat.INTEGER:
            return ColoringSolution(
                problem=self.problem,
                colors=[
                    cpsolvercb.Value(self.variables["colors"][i])
                    for i in range(len(self.variables["colors"]))
                ],
            )
        if self.modeling == ModelingCPSat.BINARY:
            colors = [None for i in range(len(self.variables["colors"]))]
            for i in range(len(self.variables["colors"])):
                c = next(
                    (
                        j
                        for j in self.variables["colors"][i]
                        if cpsolvercb.Value(self.variables["colors"][i][j]) == 1
                    ),
                    None,
                )
                colors[i] = c
            return ColoringSolution(problem=self.problem, colors=colors)

    def init_model_binary(self, nb_colors: int, **kwargs):
        cp_model = CpModel()
        allocation_binary = [
            {
                j: cp_model.NewBoolVar(name=f"allocation_{i}_{j}")
                for j in range(nb_colors)
            }
            for i in range(self.problem.number_of_nodes)
        ]
        for i in range(len(allocation_binary)):
            cp_model.AddExactlyOne(
                [allocation_binary[i][j] for j in allocation_binary[i]]
            )
        if self.problem.has_constraints_coloring:
            for node in self.problem.constraints_coloring.color_constraint:
                ind = self.problem.index_nodes_name[node]
                col = self.problem.constraints_coloring.color_constraint[node]
                cp_model.Add(allocation_binary[ind][col] == 1)
                for c in allocation_binary[ind]:
                    if c != col:
                        # Could do it more efficiently
                        cp_model.Add(allocation_binary[ind][c] == 0)
        for edge in self.problem.graph.edges:
            ind1 = self.problem.index_nodes_name[edge[0]]
            ind2 = self.problem.index_nodes_name[edge[1]]
            for team in allocation_binary[ind1]:
                if team in allocation_binary[ind2]:
                    cp_model.AddForbiddenAssignments(
                        [allocation_binary[ind1][team], allocation_binary[ind2][team]],
                        [(1, 1)],
                    )
        used = [cp_model.NewBoolVar(f"used_{j}") for j in range(nb_colors)]
        if self.problem.use_subset:
            indexes_subset = self.problem.index_subset_nodes
        else:
            indexes_subset = range(len(allocation_binary))
        for i in indexes_subset:
            for j in allocation_binary[i]:
                cp_model.Add(used[j] >= allocation_binary[i][j])
        cp_model.Minimize(sum(used))
        self.cp_model = cp_model
        self.variables["colors"] = allocation_binary
        self.variables["used"] = used

    def init_model_integer(self, nb_colors: int, **kwargs):
        used_variable = kwargs["used_variable"]
        value_sequence_chain = kwargs["value_sequence_chain"]
        symmetry_on_used = kwargs["symmetry_on_used"]
        cp_model = CpModel()
        variables = [
            cp_model.NewIntVar(0, nb_colors - 1, name=f"c_{i}")
            for i in range(self.problem.number_of_nodes)
        ]
        for edge in self.problem.graph.edges:
            ind_0 = self.problem.index_nodes_name[edge[0]]
            ind_1 = self.problem.index_nodes_name[edge[1]]
            cp_model.Add(variables[ind_0] != variables[ind_1])
        if self.problem.has_constraints_coloring:
            for node in self.problem.constraints_coloring.color_constraint:
                ind = self.problem.index_nodes_name[node]
                cp_model.Add(
                    variables[ind]
                    == self.problem.constraints_coloring.color_constraint[node]
                )
        if value_sequence_chain:
            vars = [variables[i] for i in self.problem.index_subset_nodes]
            sliding_max = [
                cp_model.NewIntVar(0, min(i, nb_colors), name=f"m_{i}")
                for i in range(len(vars))
            ]
            cp_model.Add(vars[0] == sliding_max[0])
            self.variables["sliding_max"] = sliding_max
            for k in range(1, len(vars)):
                cp_model.AddMaxEquality(sliding_max[k], [sliding_max[k - 1], vars[k]])
                cp_model.Add(sliding_max[k] <= sliding_max[k - 1] + 1)
        used = [cp_model.NewBoolVar(name=f"used_{c}") for c in range(nb_colors)]
        if used_variable:

            def add_indicator(vars, value, presence_value, model):
                bool_vars = []
                for var in vars:
                    bool_var = model.NewBoolVar("")
                    model.Add(var == value).OnlyEnforceIf(bool_var)
                    model.Add(var != value).OnlyEnforceIf(bool_var.Not())
                    bool_vars.append(bool_var)
                model.AddMaxEquality(presence_value, bool_vars)

            for j in range(nb_colors):
                if self.problem.use_subset:
                    indexes = self.problem.index_subset_nodes
                    vars = [variables[i] for i in indexes]
                else:
                    vars = variables
                add_indicator(vars, j, used[j], cp_model)
            if symmetry_on_used:
                for j in range(nb_colors - 1):
                    cp_model.Add(used[j] >= used[j + 1])
            cp_model.Minimize(sum(used))
        else:
            nbc = cp_model.NewIntVar(0, nb_colors, name="nbcolors")
            cp_model.AddMaxEquality(
                nbc, [variables[i] for i in self.problem.index_subset_nodes]
            )
            cp_model.Minimize(nbc)
        self.cp_model = cp_model
        self.variables["colors"] = variables
        self.variables["used"] = used

    def set_warm_start(self, solution: ColoringSolution) -> None:
        """Make the solver warm start from the given solution."""
        self.cp_model.clear_hints()
        if self.modeling == ModelingCPSat.INTEGER:
            self.set_warm_start_integer(solution)
        if self.modeling == ModelingCPSat.BINARY:
            self.set_warm_start_binary(solution)

    def set_warm_start_integer(self, solution: ColoringSolution):
        for i in range(len(solution.colors)):
            self.cp_model.AddHint(self.variables["colors"][i], solution.colors[i])

    def set_warm_start_binary(self, solution: ColoringSolution):
        for i in range(len(solution.colors)):
            c = solution.colors[i]
            for color in self.variables["colors"][i]:
                self.cp_model.AddHint(self.variables["colors"][i][color], color == c)

    def init_model(self, **args: Any) -> None:
        args = self.complete_with_default_hyperparameters(args)
        modeling = args["modeling"]
        do_warmstart = args["warmstart"]
        assert isinstance(modeling, ModelingCPSat)
        if "nb_colors" not in args or do_warmstart:
            solution = self.get_starting_solution(**args)
            nb_colors = self.problem.count_colors_all_index(solution.colors)
            args["nb_colors"] = min(args.get("nb_colors", nb_colors), nb_colors)
        if modeling == ModelingCPSat.BINARY:
            self.init_model_binary(**args)
        if modeling == ModelingCPSat.INTEGER:
            self.init_model_integer(**args)
        if do_warmstart:
            self.set_warm_start(solution=solution)
        self.modeling = modeling
