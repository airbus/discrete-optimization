#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from collections.abc import Iterable
from enum import Enum
from typing import Any, Optional, Union

from ortools.sat.python.cp_model import CpSolverSolutionCallback, IntVar, LinearExprT

from discrete_optimization.coloring.problem import (
    Color,
    ColoringProblem,
    ColoringSolution,
    Node,
)
from discrete_optimization.coloring.solvers.starting_solution import (
    WithStartingSolutionColoringSolver,
)
from discrete_optimization.generic_tasks_tools.solvers.cpsat import (
    AllocationBinaryOrIntegerModellingCpSatSolver,
    AllocationModelling,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
)
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    EnumHyperparameter,
)


class ModelingCpSat(Enum):
    BINARY = 0
    INTEGER = 1


class CpSatColoringSolver(
    AllocationBinaryOrIntegerModellingCpSatSolver[Node, Color],
    WithStartingSolutionColoringSolver,
    WarmstartMixin,
):
    hyperparameters = [
        EnumHyperparameter(
            name="modeling", enum=ModelingCpSat, default=ModelingCpSat.INTEGER
        ),
        CategoricalHyperparameter(
            name="do_warmstart", choices=[True, False], default=True
        ),
        CategoricalHyperparameter(
            name="value_sequence_chain",
            choices=[True, False],
            default=False,
            depends_on=("modeling", [ModelingCpSat.INTEGER]),
        ),
        CategoricalHyperparameter(
            name="used_variable",
            choices=[True, False],
            default=False,
            depends_on=("modeling", [ModelingCpSat.INTEGER]),
        ),
        CategoricalHyperparameter(
            name="symmetry_on_used",
            choices=[True, False],
            default=True,
            depends_on=("used_variable", [True]),
        ),
    ] + WithStartingSolutionColoringSolver.hyperparameters

    at_most_one_unary_resource_per_task = True
    problem: ColoringProblem
    _nb_colors: int

    def __init__(
        self,
        problem: ColoringProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.modeling: Optional[ModelingCpSat] = None
        self.variables: dict[
            str, Union[IntVar, list[IntVar], list[dict[int, IntVar]]]
        ] = {}
        self._subset_nodes = list(self.problem.subset_nodes)

    def get_binary_allocation_variable(
        self, task: Node, unary_resource: Color
    ) -> LinearExprT:
        i_task = self.problem.index_nodes_name[task]
        i_color = unary_resource
        return self.variables["colors"][i_task][i_color]

    def get_integer_allocation_variable(self, task: Node) -> LinearExprT:
        i_task = self.problem.index_nodes_name[task]
        return self.variables["colors"][i_task]

    def retrieve_solution(
        self, cpsolvercb: CpSolverSolutionCallback
    ) -> ColoringSolution:
        if self.modeling == ModelingCpSat.INTEGER:
            return ColoringSolution(
                problem=self.problem,
                colors=[
                    cpsolvercb.Value(self.variables["colors"][i])
                    for i in range(len(self.variables["colors"]))
                ],
            )
        else:  # ModelingCpSat.BINARY:
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
        self.allocation_modelling = AllocationModelling.BINARY
        super().init_model(**kwargs)
        cp_model = self.cp_model
        allocation_binary = [
            {
                j: cp_model.NewBoolVar(name=f"allocation_{i}_{j}")
                for j in range(nb_colors)
            }
            for i in range(self.problem.number_of_nodes)
        ]
        self.variables["colors"] = allocation_binary
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
        used = self.create_used_variables_list()
        cp_model.Minimize(sum(used))
        self.variables["used"] = used

    def init_model_integer(self, nb_colors: int, **kwargs):
        self.allocation_modelling = AllocationModelling.INTEGER
        used_variable = kwargs["used_variable"]
        value_sequence_chain = kwargs["value_sequence_chain"]
        if used_variable:
            symmetry_on_used = kwargs["symmetry_on_used"]
        else:
            symmetry_on_used = False
        super().init_model(**kwargs)
        cp_model = self.cp_model
        variables = [
            cp_model.NewIntVar(0, nb_colors - 1, name=f"c_{i}")
            for i in range(self.problem.number_of_nodes)
        ]
        self.variables["colors"] = variables
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
        if used_variable:
            used = self.create_used_variables_list()
            if symmetry_on_used:
                for j in range(nb_colors - 1):
                    cp_model.Add(used[j] >= used[j + 1])
            cp_model.Minimize(sum(used))
            self.variables["used"] = used
        else:
            nbc = cp_model.NewIntVar(0, nb_colors, name="nbcolors")
            self.variables["nbc"] = nbc
            cp_model.AddMaxEquality(
                nbc, [variables[i] for i in self.problem.index_subset_nodes]
            )
            cp_model.Minimize(nbc)

    def set_warm_start(self, solution: ColoringSolution) -> None:
        """Make the solver warm start from the given solution."""
        self.cp_model.clear_hints()
        if self.modeling == ModelingCpSat.INTEGER:
            self.set_warm_start_integer(solution)
        if self.modeling == ModelingCpSat.BINARY:
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
        do_warmstart = args["do_warmstart"]
        assert isinstance(modeling, ModelingCpSat)
        if "nb_colors" not in args or do_warmstart:
            solution = self.get_starting_solution(**args)
            nb_colors = self.problem.count_colors_all_index(solution.colors)
            args["nb_colors"] = min(args.get("nb_colors", nb_colors), nb_colors)
        # ensure nb_colors <= nb nodes
        args["nb_colors"] = min(args["nb_colors"], self.problem.number_of_nodes)
        self._nb_colors = args["nb_colors"]  # store nb colors max
        if modeling == ModelingCpSat.BINARY:
            self.init_model_binary(**args)
        if modeling == ModelingCpSat.INTEGER:
            self.init_model_integer(**args)
        if do_warmstart:
            self.set_warm_start(solution=solution)
        self.modeling = modeling

    @property
    def subset_tasks_of_interest(self) -> Iterable[Node]:
        return self.problem.subset_nodes

    @property
    def subset_unaryresources_allowed(self) -> Iterable[Color]:
        return range(self._nb_colors)

    def create_used_variables_list(
        self,
    ) -> list[IntVar]:
        self.create_used_variables()
        return [
            self.used_variables[color] for color in self.subset_unaryresources_allowed
        ]
