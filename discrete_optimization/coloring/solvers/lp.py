"""Linear programming models and solve functions for Coloring problem."""

#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from collections.abc import Callable, Hashable
from typing import Any, Optional, TypedDict, Union

import networkx as nx
from ortools.math_opt.python import mathopt

from discrete_optimization.coloring.problem import ColoringProblem, ColoringSolution
from discrete_optimization.coloring.solvers import ColoringSolver
from discrete_optimization.coloring.solvers.greedy import (
    GreedyColoringSolver,
    NxGreedyColoringMethod,
)
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
)
from discrete_optimization.generic_tools.lp_tools import (
    ConstraintType,
    GurobiMilpSolver,
    MilpSolver,
    OrtoolsMathOptMilpSolver,
    VariableType,
)
from discrete_optimization.generic_tools.unsat_tools import MetaConstraint

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True


logger = logging.getLogger(__name__)


OneColorConstraints = dict[int, ConstraintType]
NeighborsConstraints = dict[tuple[Hashable, Hashable, int], ConstraintType]
VariableDecision = dict[str, dict[tuple[Hashable, int], VariableType]]


class ConstraintsDict(TypedDict):
    one_color_constraints: OneColorConstraints
    constraints_neighbors: NeighborsConstraints


class _BaseLpColoringSolver(MilpSolver, ColoringSolver):
    """Base class for Coloring LP solvers."""

    hyperparameters = [
        CategoricalHyperparameter(
            name="greedy_start", choices=[True, False], default=True
        ),
        CategoricalHyperparameter(
            name="use_cliques", choices=[True, False], default=False
        ),
    ]

    def __init__(
        self,
        problem: ColoringProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.number_of_nodes = self.problem.number_of_nodes
        self.nodes_name = self.problem.graph.nodes_name
        self.index_nodes_name = {
            self.nodes_name[i]: i for i in range(self.number_of_nodes)
        }
        self.index_to_nodes_name = {
            i: self.nodes_name[i] for i in range(self.number_of_nodes)
        }
        self.graph = self.problem.graph
        self.model = None
        self.variable_decision: VariableDecision = {}
        one_color_constraints: OneColorConstraints = {}
        constraints_neighbors: NeighborsConstraints = {}
        self.constraints_dict: ConstraintsDict = {
            "one_color_constraints": one_color_constraints,
            "constraints_neighbors": constraints_neighbors,
        }
        self.description_variable_description = {
            "colors_vars": {
                "shape": (0, 0),
                "type": bool,
                "descr": "for each node and each color," " a binary indicator",
            }
        }
        self.description_constraint: dict[str, dict[str, str]] = {}
        self.sense_optim = self.params_objective_function.sense_function
        self.start_solution: Optional[ColoringSolution] = None

    def init_model(self, **kwargs: Any) -> None:
        """Initialize the internal model.

        Keyword Args:
            greedy_start (bool): if True, a greedy solution is computed (using GreedyColoring solver)
                and used as warm start for the LP.
            use_cliques (bool): if True, compute cliques of the coloring problem and add constraints to the model.

        """
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        greedy_start = kwargs["greedy_start"]
        use_cliques = kwargs["use_cliques"]
        if greedy_start:
            logger.info("Computing greedy solution")
            greedy_solver = GreedyColoringSolver(
                self.problem,
                params_objective_function=self.params_objective_function,
            )
            sol = greedy_solver.solve(
                strategy=NxGreedyColoringMethod.best
            ).get_best_solution()
            if sol is None:
                raise RuntimeError(
                    "greedy_solver.solve(strategy=NxGreedyColoringMethod.best).get_best_solution() "
                    "should not be None."
                )
            if not isinstance(sol, ColoringSolution):
                raise RuntimeError(
                    "greedy_solver.solve(strategy=NxGreedyColoringMethod.best).get_best_solution() "
                    "should be a ColoringSolution."
                )
            self.start_solution = sol
        else:
            logger.info("Get dummy solution")
            self.start_solution = self.problem.get_dummy_solution()
        nb_colors = self.start_solution.nb_color
        nb_colors_subset = nb_colors
        if self.problem.use_subset:
            nb_colors_subset = self.problem.count_colors(self.start_solution.colors)
            nb_colors = self.problem.count_colors_all_index(self.start_solution.colors)

        if nb_colors is None:
            raise RuntimeError("self.start_solution.nb_color should not be None.")
        self.model = self.create_empty_model("color")
        colors_var: dict[tuple[Hashable, int], VariableType] = {}
        range_node = self.nodes_name
        range_color = range(nb_colors)
        range_color_subset = range(nb_colors_subset)
        range_color_per_node = {}
        meta_constraints = []
        for node in self.nodes_name:
            rng = self.get_range_color(
                node_name=node,
                range_color_subset=range_color_subset,
                range_color_all=range_color,
            )
            for color in rng:
                colors_var[node, color] = self.add_binary_variable(
                    name="x_" + str((node, color))
                )
            range_color_per_node[node] = set(rng)
        one_color_constraints: OneColorConstraints = {}
        for n in range_node:
            one_color_constraints[n] = self.add_linear_constraint(
                self.construct_linear_sum(
                    colors_var[n, c] for c in range_color_per_node[n]
                )
                == 1
            )
        meta_constraints.append(
            MetaConstraint(
                name="One color per node",
                constraints=list(one_color_constraints.values()),
            )
        )
        cliques = []
        g = self.graph.to_networkx()
        if use_cliques:
            for c in nx.algorithms.clique.find_cliques(g):
                cliques += [c]
            cliques = sorted(cliques, key=lambda x: len(x), reverse=True)
        else:
            cliques = [[e[0], e[1]] for e in g.edges()]
        cliques_constraint: dict[Union[int, tuple[int, int]], ConstraintType] = {}
        index_c = 0
        opt = self.add_integer_variable(lb=0, ub=nb_colors, name="nb_colors")
        if use_cliques:
            for c in cliques[:100]:
                cliques_constraint[index_c] = self.add_linear_constraint(
                    self.construct_linear_sum(
                        (color_i + 1) * colors_var[node, color_i]
                        for node in c
                        for color_i in range_color_per_node[node]
                    )
                    >= sum([i + 1 for i in range(len(c))])
                )
                cliques_constraint[(index_c, 1)] = self.add_linear_constraint(
                    self.construct_linear_sum(
                        colors_var[node, color_i]
                        for node in c
                        for color_i in range_color_per_node[node]
                    )
                    <= opt
                )
                meta_constraints.append(
                    MetaConstraint(
                        name=f"clique({index_c})",
                        constraints=[
                            cliques_constraint[index_c],
                            cliques_constraint[(index_c, 1)],
                        ],
                    )
                )
                index_c += 1
        edges = g.edges()
        constraints_neighbors: NeighborsConstraints = {}
        meta_constraints_neighbors = [
            MetaConstraint(
                name=f"neighbours colors of node {self.problem.index_to_nodes_name[i]}"
            )
            for i in range(self.problem.number_of_nodes)
        ]
        for e in edges:
            for c in range_color_per_node[e[0]]:
                if c in range_color_per_node[e[1]]:
                    cstr = self.add_linear_constraint(
                        colors_var[e[0], c] + colors_var[e[1], c] <= 1
                    )
                    constraints_neighbors[(e[0], e[1], c)] = cstr
                    meta_constraints_neighbors[e[0]].append(cstr)
                    meta_constraints_neighbors[e[1]].append(cstr)
        meta_constraints.extend(meta_constraints_neighbors)
        opt_meta_constraint = MetaConstraint(
            name="optimality constraint",
        )
        for n in range_node:
            opt_constraint = self.add_linear_constraint(
                self.construct_linear_sum(
                    (color_i + 1) * colors_var[n, color_i]
                    for color_i in range_color_per_node[n]
                )
                <= opt
            )
            opt_meta_constraint.append(opt_constraint)
        meta_constraints.append(opt_meta_constraint)
        self.set_model_objective(opt, minimize=True)
        self.variable_decision = {"colors_var": colors_var, "nb_colors": opt}
        self.constraints_dict = {
            "one_color_constraints": one_color_constraints,
            "constraints_neighbors": constraints_neighbors,
        }
        self.description_variable_description = {
            "colors_var": {
                "shape": (self.number_of_nodes, nb_colors),
                "type": bool,
                "descr": "for each node and each color," " a binary indicator",
            }
        }
        self.description_constraint["one_color_constraints"] = {
            "descr": "one and only one color " "should be assignated to a node"
        }
        self.description_constraint["constraints_neighbors"] = {
            "descr": "no neighbors can have same color"
        }
        self.meta_constraints = meta_constraints

    def get_meta_constraints(self) -> list[MetaConstraint]:
        return self.meta_constraints

    def convert_to_variable_values(
        self, solution: ColoringSolution
    ) -> dict[Any, float]:
        """Convert a solution to a mapping between model variables and their values.

        Will be used by set_warm_start().

        """
        # Init all variables to 0
        hinted_variables = {
            var: 0 for var in self.variable_decision["colors_var"].values()
        }

        # Set var(node, color) to 1 according to the solution
        for i, color in enumerate(solution.colors):
            node = self.index_to_nodes_name[i]
            variable_decision_key = (node, color)
            hinted_variables[
                self.variable_decision["colors_var"][variable_decision_key]
            ] = 1

        hinted_variables[self.variable_decision["nb_colors"]] = solution.nb_color

        return hinted_variables

    def retrieve_current_solution(
        self,
        get_var_value_for_current_solution: Callable[[Any], float],
        get_obj_value_for_current_solution: Callable[[], float],
    ) -> ColoringSolution:
        colors = [0] * self.number_of_nodes
        for (
            variable_decision_key,
            variable_decision_value,
        ) in self.variable_decision["colors_var"].items():
            value = get_var_value_for_current_solution(variable_decision_value)
            if value >= 0.5:
                node = variable_decision_key[0]
                color = variable_decision_key[1]
                colors[self.index_nodes_name[node]] = color
        return ColoringSolution(self.problem, colors)

    def get_range_color(self, node_name, range_color_subset, range_color_all):
        if self.problem.has_constraints_coloring:
            if node_name in self.problem.constraints_coloring.nodes_fixed():
                return range(
                    self.problem.constraints_coloring.color_constraint[node_name],
                    self.problem.constraints_coloring.color_constraint[node_name] + 1,
                )
        if self.problem.use_subset:
            return (
                range_color_subset
                if node_name in self.problem.subset_nodes
                else range_color_all
            )
        else:
            return range_color_all


class GurobiColoringSolver(_BaseLpColoringSolver, GurobiMilpSolver):
    """Coloring LP solver based on gurobipy library.

    Attributes:
        problem (ColoringProblem): coloring problem instance to solve
        params_objective_function (ParamsObjectiveFunction): objective function parameters
                (however this is just used for the ResultStorage creation, not in the optimisation)

    """

    hyperparameters = _BaseLpColoringSolver.hyperparameters

    def init_model(self, **kwargs: Any) -> None:
        """Initialize the gurobi model.

        Keyword Args:
            greedy_start (bool): if True, a greedy solution is computed (using GreedyColoring solver)
                and used as warm start for the LP.
            use_cliques (bool): if True, compute cliques of the coloring problem and add constraints to the model.
        """
        _BaseLpColoringSolver.init_model(self, **kwargs)
        self.model.setParam(gurobipy.GRB.Param.Threads, 8)
        self.model.setParam(gurobipy.GRB.Param.Method, -1)
        self.model.setParam("Heuristics", 0.01)
        self.model.update()

    def convert_to_variable_values(
        self, solution: ColoringSolution
    ) -> dict[gurobipy.Var, float]:
        """Convert a solution to a mapping between model variables and their values.

        Will be used by set_warm_start().

        """
        return _BaseLpColoringSolver.convert_to_variable_values(self, solution)


class MathOptColoringSolver(_BaseLpColoringSolver, OrtoolsMathOptMilpSolver):
    """Coloring LP solver based on pymip library.

    Note:
        Gurobi and CBC are available as backend solvers.


    Attributes:
        problem (ColoringProblem): coloring problem instance to solve
        params_objective_function (ParamsObjectiveFunction): objective function parameters
                    (however this is just used for the ResultStorage creation, not in the optimisation)

    """

    hyperparameters = _BaseLpColoringSolver.hyperparameters

    problem: ColoringProblem

    def convert_to_variable_values(
        self, solution: ColoringSolution
    ) -> dict[mathopt.Variable, float]:
        """Convert a solution to a mapping between model variables and their values.

        Will be used by set_warm_start() to provide a suitable SolutionHint.variable_values.
        See https://or-tools.github.io/docs/pdoc/ortools/math_opt/python/model_parameters.html#SolutionHint
        for more information.

        """
        return _BaseLpColoringSolver.convert_to_variable_values(self, solution)
