#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import os
import time
from typing import Any, List, Optional

import clingo
from clingo import Symbol

from discrete_optimization.coloring.coloring_model import ColoringSolution
from discrete_optimization.coloring.solvers.coloring_solver_with_starting_solution import (
    SolverColoringWithStartingSolution,
)
from discrete_optimization.generic_tools.asp_tools import ASPClingoSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

cur_folder = os.path.abspath(os.path.dirname(__file__))
logger = logging.getLogger(__name__)


class ColoringASPSolver(ASPClingoSolver, SolverColoringWithStartingSolution):
    """Solver based on Answer Set Programming formulation and clingo solver."""

    hyperparameters = SolverColoringWithStartingSolution.hyperparameters

    def retrieve_solution(self, model: clingo.Model) -> ColoringSolution:
        symbols = model.symbols(atoms=True)
        colors = [
            (s.arguments[0].number, s.arguments[1].name)
            for s in symbols
            if s.name == "color"
        ]
        colors_name = list(set([s[1] for s in colors]))
        colors_to_index = self.compute_clever_colors_map(colors_name)
        colors_vect = [0] * self.problem.number_of_nodes
        for num, color in colors:
            colors_vect[num - 1] = colors_to_index[color]

        return ColoringSolution(problem=self.problem, colors=colors_vect)

    def init_model(self, **kwargs: Any) -> None:
        if self.problem.use_subset:
            self.init_model_with_subset(**kwargs)
        else:
            self.init_model_without_subset(**kwargs)

    def init_model_without_subset(self, **kwargs: Any) -> None:
        basic_model = """
        1 {color(X,C) : col(C)} 1 :- node(X).
        :- edge(X,Y), color(X,C), color(Y,C).
        color(X, C) :- fixed_color(X, C).
        #show color/2.
        ncolors(C) :- C = #count{Color: color(_,Color)}.
        #minimize {C: ncolors(C)}.
        """
        max_models = kwargs.get("max_models", 1)
        nb_colors = kwargs.get("nb_colors", None)
        if nb_colors is None:
            solution = self.get_starting_solution(
                params_objective_function=self.params_objective_function, **kwargs
            )
            nb_colors = solution.nb_color
        self.ctl = clingo.Control(
            ["--warn=no-atom-undefined", f"--models={max_models}", "--opt-mode=optN"]
        )
        self.ctl.add("base", [], basic_model)
        string_data_input = self.build_string_data_input(nb_colors=nb_colors)
        self.ctl.add("base", [], string_data_input)

    def init_model_with_subset(self, **kwargs: Any) -> None:
        basic_model = """
        1 {color(X,C) : col(C)} 1 :- node(X).
        :- edge(X,Y), color(X,C), color(Y,C).
        ncolors(C) :- C = #count{Color : color(N, Color), subset_node(N)}.
        color(X, C) :- fixed_color(X, C).
        #show color/2.
        #minimize {C: ncolors(C)}.
        """
        # # TODO make this work : :- color(X, C), subset_node(X), col(C), C > MaxValue.
        max_models = kwargs.get("max_models", 1)
        nb_colors = kwargs.get("nb_colors", None)
        nb_colors_subset = nb_colors
        if nb_colors is None:
            solution = self.get_starting_solution(
                params_objective_function=self.params_objective_function, **kwargs
            )
            nb_colors = self.problem.count_colors_all_index(solution.colors)
            nb_colors_subset = self.problem.count_colors(solution.colors)
        self.ctl = clingo.Control(
            ["--warn=no-atom-undefined", f"--models={max_models}", "--opt-mode=optN"]
        )
        self.ctl.add("base", [], basic_model)
        string_data_input = self.build_string_data_input(
            nb_colors=nb_colors, nb_colors_subset=nb_colors_subset
        )
        self.ctl.add("base", [], string_data_input)

    def build_string_data_input(
        self, nb_colors, nb_colors_subset: Optional[int] = None
    ):
        if nb_colors_subset is None:
            nb_colors_subset = nb_colors
        number_of_nodes = self.problem.number_of_nodes
        nodes = f"node(1..{number_of_nodes}).\n"
        edges = ""
        index_nodes_name = self.problem.index_nodes_name
        for e in self.problem.graph.edges:
            edges += f"edge({index_nodes_name[e[0]]+1}, {index_nodes_name[e[1]]+1}). "
        types = ""
        if self.problem.use_subset:
            # TODO : make this work.
            # types += f"max_value({nb_colors_subset}). \n"
            for node in self.problem.subset_nodes:
                if node in self.problem.subset_nodes:
                    types += f"subset_node({self.problem.index_nodes_name[node] + 1}). "
        constraints = ""
        if self.problem.constraints_coloring:
            constraints = self.constrained_data_input()
        colors = " ".join([f"col(c_{i})." for i in range(nb_colors)])
        full_string_input = (
            nodes + edges + "\n" + colors + "\n" + types + "\n" + constraints + "\n"
        )
        return full_string_input

    def constrained_data_input(self):
        s = ""
        if self.problem.constraints_coloring.color_constraint is not None:
            for node in self.problem.constraints_coloring.color_constraint:
                value = self.problem.constraints_coloring.color_constraint[node]
                index_node = self.problem.index_nodes_name[node]
                s += f"fixed_color({index_node+1}, c_{value}). "
        return s

    def compute_clever_colors_map(self, colors_name: List[str]):
        colors_to_protect = set()
        colors_to_index = {}
        if self.problem.has_constraints_coloring:
            colors_to_protect = set(
                [
                    f"c_{x}"
                    for x in self.problem.constraints_coloring.color_constraint.values()
                ]
            )
            colors_to_index = {
                f"c_{x}": x
                for x in self.problem.constraints_coloring.color_constraint.values()
            }
        color_name = [
            colors_name[j]
            for j in range(len(colors_name))
            if colors_name[j] not in colors_to_protect
        ]
        value_name = [
            j
            for j in range(len(colors_name))
            if j not in [v for v in colors_to_index.values()]
        ]
        for c, val in zip(color_name, value_name):
            colors_to_index[c] = val
        return colors_to_index
