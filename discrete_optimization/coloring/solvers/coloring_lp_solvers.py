"""Linear programming models and solve functions for Coloring problem."""

#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import sys
from typing import Any, Dict, Hashable, List, Optional, Tuple, Union

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
    Solution,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.lp_tools import (
    GurobiMilpSolver,
    MilpSolver,
    MilpSolverName,
    ParametersMilp,
    PymipMilpSolver,
    map_solver,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
    TupleFitness,
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

    def __init__(
        self,
        coloring_model: ColoringProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        SolverColoring.__init__(self, coloring_model=coloring_model)
        self.number_of_nodes = self.coloring_model.number_of_nodes
        self.nodes_name = self.coloring_model.graph.nodes_name
        self.index_nodes_name = {
            self.nodes_name[i]: i for i in range(self.number_of_nodes)
        }
        self.index_to_nodes_name = {
            i: self.nodes_name[i] for i in range(self.number_of_nodes)
        }
        self.graph = self.coloring_model.graph
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
        (
            self.aggreg_from_sol,
            self.aggreg_from_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.coloring_model,
            params_objective_function=params_objective_function,
        )
        self.sense_optim = self.params_objective_function.sense_function
        self.start_solution: Optional[ColoringSolution] = None

    def retrieve_solutions(self, parameters_milp: ParametersMilp) -> ResultStorage:
        if parameters_milp.retrieve_all_solution:
            n_solutions = min(parameters_milp.n_solutions_max, self.nb_solutions)
        else:
            n_solutions = 1
        list_solution_fits: List[Tuple[Solution, Union[float, TupleFitness]]] = []
        for s in range(n_solutions):
            colors = [0] * self.number_of_nodes
            for (
                variable_decision_key,
                variable_decision_value,
            ) in self.variable_decision["colors_var"].items():
                value = self.get_var_value_for_ith_solution(variable_decision_value, s)
                if value >= 0.5:
                    node = variable_decision_key[0]
                    color = variable_decision_key[1]
                    colors[self.index_nodes_name[node]] = color
            solution = ColoringSolution(self.coloring_model, colors)
            fit = self.aggreg_from_sol(solution)
            list_solution_fits.append((solution, fit))
        return ResultStorage(
            list_solution_fits=list_solution_fits,
            mode_optim=self.sense_optim,
        )


class ColoringLP(GurobiMilpSolver, _BaseColoringLP):
    """Coloring LP solver based on gurobipy library.

    Attributes:
        coloring_model (ColoringProblem): coloring problem instance to solve
        params_objective_function (ParamsObjectiveFunction): objective function parameters
                (however this is just used for the ResultStorage creation, not in the optimisation)

    """

    def init_model(self, **kwargs: Any) -> None:
        """Initialize the gurobi model.

        Keyword Args:
            greedy_start (bool): if True, a greedy solution is computed (using GreedyColoring solver)
                and used as warm start for the LP.
            use_cliques (bool): if True, compute cliques of the coloring problem and add constraints to the model.
            verbose (bool): verbose option.
        """
        greedy_start = kwargs.get("greedy_start", True)
        use_cliques = kwargs.get("use_cliques", False)
        if greedy_start:
            logger.info("Computing greedy solution")
            greedy_solver = GreedyColoring(
                self.coloring_model,
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
            self.start_solution = self.coloring_model.get_dummy_solution()
        nb_colors = self.start_solution.nb_color
        if nb_colors is None:
            raise RuntimeError("self.start_solution.nb_color should not be None.")
        color_model = Model("color")
        colors_var: Dict[Tuple[Hashable, int], "Var"] = {}
        range_node = range(self.number_of_nodes)
        range_color = range(nb_colors)
        for node in self.nodes_name:
            for color in range_color:
                colors_var[node, color] = color_model.addVar(
                    vtype=GRB.BINARY, obj=0, name="x_" + str((node, color))
                )
        one_color_constraints: OneColorConstraints = {}
        for n in range_node:
            one_color_constraints[n] = color_model.addConstr(
                quicksum([colors_var[n, c] for c in range_color]) == 1
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
                cliques_constraint[index_c] = color_model.addConstr(
                    quicksum(
                        [
                            (color_i + 1) * colors_var[node, color_i]
                            for node in c
                            for color_i in range_color
                        ]
                    )
                    >= sum([i + 1 for i in range(len(c))])
                )
                cliques_constraint[(index_c, 1)] = color_model.addConstr(
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
            for c in range_color:
                constraints_neighbors[(e[0], e[1], c)] = color_model.addConstr(
                    colors_var[e[0], c] + colors_var[e[1], c] <= 1
                )
        for n in range_node:
            color_model.addConstr(
                quicksum(
                    [(color_i + 1) * colors_var[n, color_i] for color_i in range_color]
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

    def retrieve_solutions(self, parameters_milp: ParametersMilp) -> ResultStorage:
        # We call explicitely the method to be sure getting the proper one
        return _BaseColoringLP.retrieve_solutions(self, parameters_milp=parameters_milp)


class ColoringLP_MIP(PymipMilpSolver, _BaseColoringLP):
    """Coloring LP solver based on pymip library.

    Note:
        Gurobi and CBC are available as backend solvers.


    Attributes:
        coloring_model (ColoringProblem): coloring problem instance to solve
        params_objective_function (ParamsObjectiveFunction): objective function parameters
                    (however this is just used for the ResultStorage creation, not in the optimisation)
        milp_solver_name (MilpSolverName): backend solver to use (either CBC ou GRB)

    """

    def __init__(
        self,
        coloring_model: ColoringProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        milp_solver_name: MilpSolverName = MilpSolverName.CBC,
        **kwargs: Any,
    ):
        super().__init__(
            coloring_model=coloring_model,
            params_objective_function=params_objective_function,
            **kwargs,
        )
        self.milp_solver_name = milp_solver_name
        self.solver_name = map_solver[milp_solver_name]

    def init_model(self, **kwargs: Any) -> None:
        greedy_start = kwargs.get("greedy_start", True)
        use_cliques = kwargs.get("use_cliques", False)
        if greedy_start:
            logger.info("Computing greedy solution")
            greedy_solver = GreedyColoring(
                self.coloring_model,
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
            self.start_solution = self.coloring_model.get_dummy_solution()
        nb_colors = self.start_solution.nb_color
        if nb_colors is None:
            raise RuntimeError("self.start_solution.nb_color should not be None.")
        color_model = mip.Model(
            "color", sense=mip.MINIMIZE, solver_name=self.solver_name
        )
        colors_var = {}
        range_node = range(self.number_of_nodes)
        range_color = range(nb_colors)
        for node in self.nodes_name:
            for color in range_color:
                colors_var[node, color] = color_model.add_var(
                    var_type=BINARY, obj=0, name="x_" + str((node, color))
                )
        one_color_constraints = {}
        for n in range_node:
            one_color_constraints[n] = color_model.add_constr(
                xsum([colors_var[n, c] for c in range_color]) == 1
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
                            for color_i in range_color
                        ]
                    )
                    >= sum([i + 1 for i in range(len(c))])
                )
                cliques_constraint[(index_c, 1)] = color_model.add_constr(
                    xsum(
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
        constraints_neighbors = {}
        for e in edges:
            for c in range_color:
                constraints_neighbors[(e[0], e[1], c)] = color_model.add_constr(
                    colors_var[e[0], c] + colors_var[e[1], c] <= 1
                )
        for n in range_node:
            color_model.add_constr(
                xsum(
                    [(color_i + 1) * colors_var[n, color_i] for color_i in range_color]
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

    def retrieve_solutions(self, parameters_milp: ParametersMilp) -> ResultStorage:
        # We call explicitely the method to be sure getting the proper one
        return _BaseColoringLP.retrieve_solutions(self, parameters_milp=parameters_milp)
