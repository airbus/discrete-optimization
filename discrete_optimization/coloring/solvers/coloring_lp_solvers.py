"""Linear programming models and solve functions for Coloring problem."""

#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import sys
from typing import Any, Callable, Dict, Hashable, Optional, Tuple, Union

import mip
import networkx as nx
from mip import BINARY, INTEGER, xsum

from discrete_optimization.coloring.coloring_model import (
    ColoringProblem,
    ColoringSolution,
)
from discrete_optimization.coloring.solvers.coloring_solver import SolverColoring
from discrete_optimization.coloring.solvers.greedy_coloring import (
    GreedyColoring,
    NXGreedyColoringMethod,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Problem,
    Solution,
)
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
)
from discrete_optimization.generic_tools.lp_tools import (
    GurobiMilpSolver,
    MilpSolver,
    MilpSolverName,
    PymipMilpSolver,
)

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True
    from gurobipy import GRB, Constr, GenConstr, MConstr, Model, QConstr, Var, quicksum

if sys.version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict


logger = logging.getLogger(__name__)


OneColorConstraints = Dict[int, Union["Constr", "QConstr", "MConstr", "GenConstr"]]
NeighborsConstraints = Dict[
    Tuple[Hashable, Hashable, int], Union["Constr", "QConstr", "MConstr", "GenConstr"]
]
VariableDecision = Dict[str, Dict[Tuple[Hashable, int], Union["Var", mip.Var]]]


class ConstraintsDict(TypedDict):
    one_color_constraints: OneColorConstraints
    constraints_neighbors: NeighborsConstraints


class _BaseColoringLP(MilpSolver, SolverColoring):
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
        self.description_constraint: Dict[str, Dict[str, str]] = {}
        self.sense_optim = self.params_objective_function.sense_function
        self.start_solution: Optional[ColoringSolution] = None

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


class ColoringLP(GurobiMilpSolver, _BaseColoringLP, WarmstartMixin):
    """Coloring LP solver based on gurobipy library.

    Attributes:
        problem (ColoringProblem): coloring problem instance to solve
        params_objective_function (ParamsObjectiveFunction): objective function parameters
                (however this is just used for the ResultStorage creation, not in the optimisation)

    """

    hyperparameters = _BaseColoringLP.hyperparameters

    def init_model(self, **kwargs: Any) -> None:
        """Initialize the gurobi model.

        Keyword Args:
            greedy_start (bool): if True, a greedy solution is computed (using GreedyColoring solver)
                and used as warm start for the LP.
            use_cliques (bool): if True, compute cliques of the coloring problem and add constraints to the model.
            verbose (bool): verbose option.
        """
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        greedy_start = kwargs["greedy_start"]
        use_cliques = kwargs["use_cliques"]
        if greedy_start:
            logger.info("Computing greedy solution")
            greedy_solver = GreedyColoring(
                self.problem,
                params_objective_function=self.params_objective_function,
            )
            sol = greedy_solver.solve(
                strategy=NXGreedyColoringMethod.best
            ).get_best_solution()
            if sol is None:
                raise RuntimeError(
                    "greedy_solver.solve(strategy=NXGreedyColoringMethod.best).get_best_solution() "
                    "should not be None."
                )
            if not isinstance(sol, ColoringSolution):
                raise RuntimeError(
                    "greedy_solver.solve(strategy=NXGreedyColoringMethod.best).get_best_solution() "
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
        color_model = Model("color")
        colors_var: Dict[Tuple[Hashable, int], "Var"] = {}
        range_node = self.nodes_name
        range_color = range(nb_colors)
        range_color_subset = range(nb_colors_subset)
        range_color_per_node = {}
        for node in self.nodes_name:
            rng = self.get_range_color(
                node_name=node,
                range_color_subset=range_color_subset,
                range_color_all=range_color,
            )
            for color in rng:
                colors_var[node, color] = color_model.addVar(
                    vtype=GRB.BINARY, obj=0, name="x_" + str((node, color))
                )
            range_color_per_node[node] = set(rng)
        one_color_constraints: OneColorConstraints = {}
        for n in range_node:
            one_color_constraints[n] = color_model.addLConstr(
                quicksum([colors_var[n, c] for c in range_color_per_node[n]]) == 1
            )
        color_model.update()
        cliques = []
        g = self.graph.to_networkx()
        if use_cliques:
            for c in nx.algorithms.clique.find_cliques(g):
                cliques += [c]
            cliques = sorted(cliques, key=lambda x: len(x), reverse=True)
        else:
            cliques = [[e[0], e[1]] for e in g.edges()]
        cliques_constraint: Dict[Union[int, Tuple[int, int]], Any] = {}
        index_c = 0
        opt = color_model.addVar(vtype=GRB.INTEGER, lb=0, ub=nb_colors, obj=1)
        if use_cliques:
            for c in cliques[:100]:
                cliques_constraint[index_c] = color_model.addLConstr(
                    quicksum(
                        [
                            (color_i + 1) * colors_var[node, color_i]
                            for node in c
                            for color_i in range_color_per_node[node]
                        ]
                    )
                    >= sum([i + 1 for i in range(len(c))])
                )
                cliques_constraint[(index_c, 1)] = color_model.addLConstr(
                    quicksum(
                        [
                            colors_var[node, color_i]
                            for node in c
                            for color_i in range_color
                        ]
                    )
                    <= opt
                )
                index_c += 1
        edges = g.edges()
        constraints_neighbors: NeighborsConstraints = {}
        for e in edges:
            for c in range_color_per_node[e[0]]:
                if c in range_color_per_node[e[1]]:
                    constraints_neighbors[(e[0], e[1], c)] = color_model.addLConstr(
                        colors_var[e[0], c] + colors_var[e[1], c] <= 1
                    )
        for n in range_node:
            color_model.addLConstr(
                quicksum(
                    [
                        (color_i + 1) * colors_var[n, color_i]
                        for color_i in range_color_per_node[n]
                    ]
                )
                <= opt
            )
        color_model.update()
        color_model.modelSense = GRB.MINIMIZE
        color_model.setParam(GRB.Param.Threads, 8)
        color_model.setParam(GRB.Param.PoolSolutions, 10000)
        color_model.setParam(GRB.Param.Method, -1)
        color_model.setParam("MIPGapAbs", 0.001)
        color_model.setParam("MIPGap", 0.001)
        color_model.setParam("Heuristics", 0.01)
        self.model = color_model
        self.variable_decision = {"colors_var": colors_var}
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

    def set_warm_start(self, solution: ColoringSolution) -> None:
        """Make the solver warm start from the given solution."""
        # Init all variables to 0
        for var in self.variable_decision["colors_var"].values():
            var.Start = 0
        # Set var(node, color) to 1 according to the solution
        for i, color in enumerate(solution.colors):
            node = self.index_to_nodes_name[i]
            variable_decision_key = (node, color)
            self.variable_decision["colors_var"][variable_decision_key].Start = 1


class ColoringLP_MIP(PymipMilpSolver, _BaseColoringLP):
    """Coloring LP solver based on pymip library.

    Note:
        Gurobi and CBC are available as backend solvers.


    Attributes:
        problem (ColoringProblem): coloring problem instance to solve
        params_objective_function (ParamsObjectiveFunction): objective function parameters
                    (however this is just used for the ResultStorage creation, not in the optimisation)
        milp_solver_name (MilpSolverName): backend solver to use (either CBC ou GRB)

    """

    hyperparameters = _BaseColoringLP.hyperparameters

    problem: ColoringProblem

    def __init__(
        self,
        problem: ColoringProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        milp_solver_name: MilpSolverName = MilpSolverName.CBC,
        **kwargs: Any,
    ):
        _BaseColoringLP.__init__(
            self,
            problem=problem,
            params_objective_function=params_objective_function,
            **kwargs,
        )
        self.set_milp_solver_name(milp_solver_name=milp_solver_name)

    def init_model(self, **kwargs: Any) -> None:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        greedy_start = kwargs["greedy_start"]
        use_cliques = kwargs["use_cliques"]
        if greedy_start:
            logger.info("Computing greedy solution")
            greedy_solver = GreedyColoring(
                self.problem,
                params_objective_function=self.params_objective_function,
            )
            sol = greedy_solver.solve(
                strategy=NXGreedyColoringMethod.best
            ).get_best_solution()
            if sol is None:
                raise RuntimeError(
                    "greedy_solver.solve(strategy=NXGreedyColoringMethod.best).get_best_solution() "
                    "should not be None."
                )
            if not isinstance(sol, ColoringSolution):
                raise RuntimeError(
                    "greedy_solver.solve(strategy=NXGreedyColoringMethod.best).get_best_solution() "
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
        color_model = mip.Model(
            "color", sense=mip.MINIMIZE, solver_name=self.solver_name
        )
        colors_var = {}
        range_node = self.nodes_name
        range_color = range(nb_colors)
        range_color_subset = range(nb_colors_subset)
        range_color_per_node = {}
        for node in self.nodes_name:
            rng = self.get_range_color(
                node_name=node,
                range_color_subset=range_color_subset,
                range_color_all=range_color,
            )
            for color in rng:
                colors_var[node, color] = color_model.add_var(
                    var_type=BINARY, obj=0, name="x_" + str((node, color))
                )
            range_color_per_node[node] = set(rng)
        one_color_constraints = {}
        for n in range_node:
            one_color_constraints[n] = color_model.add_constr(
                xsum([colors_var[n, c] for c in range_color_per_node[n]]) == 1
            )
        cliques = []
        g = self.graph.to_networkx()
        if use_cliques:
            for c in nx.algorithms.clique.find_cliques(g):
                cliques += [c]
            cliques = sorted(cliques, key=lambda x: len(x), reverse=True)
        else:
            cliques = [[e[0], e[1]] for e in g.edges()]
        cliques_constraint: Dict[Union[int, Tuple[int, int]], Any] = {}
        index_c = 0
        opt = color_model.add_var(var_type=INTEGER, lb=0, ub=nb_colors, obj=1)
        if use_cliques:
            for c in cliques[:100]:
                cliques_constraint[index_c] = color_model.add_constr(
                    xsum(
                        [
                            (color_i + 1) * colors_var[node, color_i]
                            for node in c
                            for color_i in range_color_per_node[node]
                        ]
                    )
                    >= sum([i + 1 for i in range(len(c))])
                )
                cliques_constraint[(index_c, 1)] = color_model.add_constr(
                    xsum(
                        [
                            colors_var[node, color_i]
                            for node in c
                            for color_i in range_color_per_node[node]
                        ]
                    )
                    <= opt
                )
                index_c += 1
        edges = g.edges()
        constraints_neighbors = {}
        for e in edges:
            for c in range_color_per_node[e[0]]:
                if c in range_color_per_node[e[1]]:
                    constraints_neighbors[(e[0], e[1], c)] = color_model.add_constr(
                        colors_var[e[0], c] + colors_var[e[1], c] <= 1
                    )
        for n in range_node:
            color_model.add_constr(
                xsum(
                    [
                        (color_i + 1) * colors_var[n, color_i]
                        for color_i in range_color_per_node[n]
                    ]
                )
                <= opt
            )
        self.model = color_model
        self.variable_decision = {"colors_var": colors_var}
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
