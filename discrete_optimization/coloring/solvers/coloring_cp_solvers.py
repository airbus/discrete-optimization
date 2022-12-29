"""Module containing Constraint Programming based solver for Coloring Problem.

CP formulation rely on minizinc models stored in coloring/minizinc folder.
"""

#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import os
import random
from datetime import timedelta
from enum import Enum
from typing import Any, Iterable, List, Optional, Tuple

import networkx as nx
import pymzn
from minizinc import Instance, Model, Result, Solver

from discrete_optimization.coloring.coloring_model import (
    ColoringProblem,
    ColoringSolution,
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

path_minizinc = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../minizinc/")
)

logger = logging.getLogger(__name__)


class ColoringCPSolution:
    objective: int
    __output_item: Optional[str] = None

    def __init__(self, objective: int, _output_item: Optional[str], **kwargs: Any):
        self.objective = objective
        self.dict = kwargs
        logger.debug(f"New solution {self.objective}")
        logger.debug(f"Output {_output_item}")

    def check(self) -> bool:
        return True


class ColoringCPModel(Enum):
    CLIQUES = 0
    DEFAULT = 1
    LNS = 2


file_dict = {
    ColoringCPModel.CLIQUES: "coloring_clique.mzn",
    ColoringCPModel.DEFAULT: "coloring.mzn",
    ColoringCPModel.LNS: "coloring_for_lns.mzn",
}


class ColoringCP(MinizincCPSolver, SolverColoring):
    def __init__(
        self,
        coloring_model: ColoringProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        cp_solver_name: CPSolverName = CPSolverName.CHUFFED,
        silent_solve_error: bool = False,
        **kwargs: Any,
    ):
        """CP solver linked with minizinc implementation of coloring problem.

        Args:
            coloring_model (ColoringProblem): coloring problem instance to solve
            params_objective_function (ParamsObjectiveFunction): params of the objective function
            cp_solver_name (CPSolverName): backend solver to use with minizinc
            silent_solve_error: if True, raise a warning instead of an error if the underlying instance.solve() crashes
            **args:
        """
        SolverColoring.__init__(self, coloring_model=coloring_model)
        self.silent_solve_error = silent_solve_error
        self.number_of_nodes = self.coloring_model.number_of_nodes
        self.number_of_edges = len(self.coloring_model.graph.edges_infos_dict)
        self.nodes_name = self.coloring_model.graph.nodes_name
        self.index_nodes_name = self.coloring_model.index_nodes_name
        self.index_to_nodes_name = self.coloring_model.index_to_nodes_name
        self.graph = self.coloring_model.graph
        self.custom_output_type = False
        self.g = None
        (
            self.aggreg_sol,
            self.aggreg_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.coloring_model,
            params_objective_function=params_objective_function,
        )
        self.cp_solver_name = cp_solver_name

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
        nb_colors = kwargs.get("nb_colors", None)
        object_output = kwargs.get("object_output", True)
        include_seq_chain_constraint = kwargs.get("include_seq_chain_constraint", False)
        if nb_colors is None:
            solution = self.get_solution(**kwargs)
            nb_colors = solution.nb_color
        model_type = kwargs.get("cp_model", ColoringCPModel.DEFAULT)
        path = os.path.join(path_minizinc, file_dict[model_type])
        self.model = Model(path)
        if object_output:
            self.model.output_type = ColoringCPSolution
            self.custom_output_type = True
        solver = Solver.lookup(map_cp_solver_name[self.cp_solver_name])
        instance = Instance(solver, self.model)
        instance["n_nodes"] = self.number_of_nodes
        instance["n_edges"] = int(self.number_of_edges / 2)
        instance["nb_colors"] = nb_colors
        keys = []
        if model_type in {ColoringCPModel.DEFAULT, ColoringCPModel.CLIQUES}:
            instance["include_seq_chain_constraint"] = include_seq_chain_constraint
            keys += ["include_seq_chain_constraint"]
        keys += ["n_nodes", "n_edges", "nb_colors"]
        edges = [
            [self.index_nodes_name[e[0]] + 1, self.index_nodes_name[e[1]] + 1, e[2]]
            for e in self.coloring_model.graph.edges
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

    def retrieve_solutions(
        self, result: Result, parameters_cp: ParametersCP
    ) -> ResultStorage:
        """Retrieve the solution found by solving the minizinc instance

        Args:
            result: result of solve() call on minizinc instance
            parameters_cp (ParametersCP): parameters of the cp solving, to specify notably how much solution is expected.

        Returns (ResultStorage): result object storing the solutions found by the CP solver.

        """
        intermediate_solutions = parameters_cp.intermediate_solution
        colors = []
        objectives = []
        solutions_fit: List[Tuple[Solution, fitness_class]] = []
        if intermediate_solutions:
            for i in range(len(result)):
                if not self.custom_output_type:
                    colors.append(result[i, "color_graph"])
                    objectives.append(result[i, "objective"])
                else:
                    colors.append(result[i].dict["color_graph"])
                    objectives.append(result[i].objective)
        else:
            if not self.custom_output_type:
                colors.append(result["color_graph"])
                objectives.append(result["objective"])
            else:
                colors.append(result.dict["color_graph"])
                objectives.append(result.objective)
        for k in range(len(colors)):
            sol = [
                colors[k][self.index_nodes_name[self.nodes_name[i]]] - 1
                for i in range(self.number_of_nodes)
            ]
            color_sol = ColoringSolution(self.coloring_model, sol)
            fit = self.aggreg_sol(color_sol)
            solutions_fit.append((color_sol, fit))

        return ResultStorage(
            list_solution_fits=solutions_fit,
            limit_store=False,
            mode_optim=self.params_objective_function.sense_function,
        )

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
                self.coloring_model,
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
            solution = self.coloring_model.get_dummy_solution()
        return solution
