"""Module containing Constraint Programming based solver for Coloring Problem.

CP formulation rely on minizinc models stored in coloring/minizinc folder.
"""

#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import os
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

import networkx as nx
import pymzn
from minizinc import Instance, Model, Solver

from discrete_optimization.coloring.coloring_model import (
    ColoringProblem,
    ColoringSolution,
    ConstraintsColoring,
)
from discrete_optimization.coloring.coloring_toolbox import compute_cliques
from discrete_optimization.coloring.solvers.coloring_solver import SolverColoring
from discrete_optimization.coloring.solvers.greedy_coloring import (
    GreedyColoring,
    NXGreedyColoringMethod,
)
from discrete_optimization.generic_tools.cp_tools import (
    CPSolverName,
    MinizincCPSolver,
    find_right_minizinc_solver_name,
)
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    EnumHyperparameter,
)

path_minizinc = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../minizinc/")
)

logger = logging.getLogger(__name__)


class ColoringCPModel(Enum):
    CLIQUES = 0
    DEFAULT = 1
    LNS = 2
    DEFAULT_WITH_SUBSET = 3


file_dict = {
    ColoringCPModel.CLIQUES: "coloring_clique.mzn",
    ColoringCPModel.DEFAULT: "coloring.mzn",
    ColoringCPModel.LNS: "coloring_for_lns.mzn",
    ColoringCPModel.DEFAULT_WITH_SUBSET: "coloring_subset_nodes.mzn",
}


class ColoringCP(MinizincCPSolver, SolverColoring):
    hyperparameters = [
        EnumHyperparameter(
            name="cp_model", enum=ColoringCPModel, default=ColoringCPModel.DEFAULT
        ),
        CategoricalHyperparameter(
            name="include_seq_chain_constraint", choices=[True, False], default=True
        ),
    ]

    def __init__(
        self,
        problem: ColoringProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        cp_solver_name: CPSolverName = CPSolverName.CHUFFED,
        silent_solve_error: bool = False,
        **kwargs: Any,
    ):
        """CP solver linked with minizinc implementation of coloring problem.

        Args:
            problem (ColoringProblem): coloring problem instance to solve
            params_objective_function (ParamsObjectiveFunction): params of the objective function
            cp_solver_name (CPSolverName): backend solver to use with minizinc
            silent_solve_error: if True, raise a warning instead of an error if the underlying instance.solve() crashes
            **args:
        """
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.silent_solve_error = silent_solve_error
        self.number_of_nodes = self.problem.number_of_nodes
        self.number_of_edges = len(self.problem.graph.edges_infos_dict)
        self.nodes_name = self.problem.graph.nodes_name
        self.index_nodes_name = self.problem.index_nodes_name
        self.index_to_nodes_name = self.problem.index_to_nodes_name
        self.graph = self.problem.graph
        self.g = None
        self.cp_solver_name = cp_solver_name
        self.dict_datas: Optional[Dict[str, Any]] = None

    def init_model(self, **kwargs: Any) -> None:
        """Instantiate a minizinc model with the coloring problem data.

        Keyword Args:
            nb_colors (int): upper bound of number of colors to be considered by the model.
            object_output (bool): specify if the solution are returned in a ColoringCPSolution object
                                  or native minizinc output.
            include_seq_chain_constraint (bool) : include the value_precede_chain in the minizinc model.
                        See documentation of minizinc for the specification of this global constraint.
            cp_model (ColoringCPModel): CP model version.
            max_cliques (int): if cp_model == ColoringCPModel.CLIQUES, specify the max number of cliques to include
                               in the model.


        Returns: None
        """
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        nb_colors = kwargs.get("nb_colors", None)
        object_output = kwargs.get("object_output", True)
        with_subset_nodes = kwargs.get("with_subset_nodes", self.problem.use_subset)
        include_seq_chain_constraint = kwargs["include_seq_chain_constraint"]

        if nb_colors is None:
            solution: ColoringSolution = self.get_solution(**kwargs)
            nb_colors = self.problem.count_colors_all_index(solution.colors)
        model_type = kwargs["cp_model"]
        if with_subset_nodes and model_type != ColoringCPModel.DEFAULT_WITH_SUBSET:
            model_type = ColoringCPModel.DEFAULT_WITH_SUBSET
        path = os.path.join(path_minizinc, file_dict[model_type])
        model = Model(path)
        solver = Solver.lookup(find_right_minizinc_solver_name(self.cp_solver_name))
        instance = Instance(solver, model)
        instance["n_nodes"] = self.number_of_nodes
        instance["n_edges"] = int(self.number_of_edges / 2)
        instance["nb_colors"] = nb_colors
        keys = []
        if model_type in {
            ColoringCPModel.DEFAULT,
            ColoringCPModel.CLIQUES,
            ColoringCPModel.DEFAULT_WITH_SUBSET,
        }:
            instance["include_seq_chain_constraint"] = include_seq_chain_constraint
            keys += ["include_seq_chain_constraint"]
        if model_type == ColoringCPModel.DEFAULT_WITH_SUBSET:
            instance["subset_node"] = [
                node in self.problem.subset_nodes for node in self.problem.nodes_name
            ]
            keys += ["subset_node"]
        keys += ["n_nodes", "n_edges", "nb_colors"]
        edges = [
            [self.index_nodes_name[e[0]] + 1, self.index_nodes_name[e[1]] + 1, e[2]]
            for e in self.problem.graph.edges
        ]
        g = nx.Graph()
        g.add_nodes_from([i for i in range(1, self.number_of_nodes + 1)])
        g.add_edges_from(edges)
        self.g = g
        if model_type == ColoringCPModel.CLIQUES:
            cliques, not_all = compute_cliques(g, kwargs.get("max_cliques", 200))
            instance["cliques"] = [set(c) for c in cliques]
            instance["n_cliques"] = len(instance["cliques"])
            instance["all_cliques"] = not not_all
            keys += ["cliques", "n_cliques", "all_cliques"]
        instance["list_edges"] = [[e[0], e[1]] for e in edges]
        keys += ["list_edges"]
        if self.problem.has_constraints_coloring:
            constraints = self.add_coloring_constraint(
                coloring_constraint=self.problem.constraints_coloring
            )
            for c in constraints:
                instance.add_string(c)
        self.instance = instance
        self.dict_datas = {k: instance[k] for k in keys}

    def export_dzn(
        self, file_name: Optional[str] = None, keys: Optional[Iterable[Any]] = None
    ) -> None:
        """[DEBUG utility] Export the instantiated data into a dzn for potential debugs without python.

        Args:
            file_name (str): file path where to dump the data file
            keys (List[str]): list of input data names to dump.

        Returns: None

        """
        if file_name is None:
            file_name = os.path.join(path_minizinc, "coloring_example_dzn.dzn")
        if keys is None:
            keys = list(self.dict_datas.keys())
        pymzn.dict2dzn(
            {k: self.dict_datas[k] for k in keys if k in self.dict_datas},
            fout=file_name,
        )
        logger.info(f"Successfully dumped data file {file_name}")

    def retrieve_solution(
        self, _output_item: Optional[str] = None, **kwargs: Any
    ) -> ColoringSolution:
        """Return a d-o solution from the variables computed by minizinc.

        Args:
            _output_item: string representing the minizinc solver output passed by minizinc to the solution constructor
            **kwargs: keyword arguments passed by minzinc to the solution contructor
                containing the objective value (key "objective"),
                and the computed variables as defined in minizinc model.

        Returns:

        """
        colors = kwargs["color_graph"]
        sol = [
            colors[self.index_nodes_name[self.nodes_name[i]]] - 1
            for i in range(self.number_of_nodes)
        ]
        return ColoringSolution(self.problem, sol)

    def add_coloring_constraint(self, coloring_constraint: ConstraintsColoring):
        s = []
        for n in coloring_constraint.color_constraint:
            index = self.index_nodes_name[n]
            value_color = coloring_constraint.color_constraint[n]
            s.append(f"constraint color_graph[{index+1}]=={value_color+1};\n")
        return s

    def get_solution(self, **kwargs: Any) -> ColoringSolution:
        """Used by the init_model method to provide a greedy first solution

        Keyword Args:
            greedy_start (bool): use heuristics (based on networkx) to compute starting solution, otherwise the
                          dummy method is used.
            verbose (bool): verbose option.

        Returns (ColoringSolution): a starting coloring solution that can be used by lns.

        """
        greedy_start = kwargs.get("greedy_start", True)
        if greedy_start:
            logger.info("Computing greedy solution")
            greedy_solver = GreedyColoring(
                self.problem,
                params_objective_function=self.params_objective_function,
            )
            result_store = greedy_solver.solve(
                strategy=kwargs.get("greedy_method", NXGreedyColoringMethod.best),
            )
            solution = result_store.get_best_solution()
            if solution is None:
                raise RuntimeError(
                    "greedy_solver.solve().get_best_solution() " "should not be None."
                )
            if not isinstance(solution, ColoringSolution):
                raise RuntimeError(
                    "greedy_solver.solve().get_best_solution() "
                    "should be a ColoringSolution."
                )
        else:
            logger.info("Get dummy solution")
            solution = self.problem.get_dummy_solution()
        return solution
