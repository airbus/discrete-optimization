#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import os
from typing import Any

import clingo

from discrete_optimization.generic_tools.asp_tools import AspClingoSolver
from discrete_optimization.maximum_independent_set.problem import (
    MisProblem,
    MisSolution,
)
from discrete_optimization.maximum_independent_set.solvers.mis_solver import MisSolver

cur_folder = os.path.abspath(os.path.dirname(__file__))
logger = logging.getLogger(__name__)


class AspMisSolver(MisSolver, AspClingoSolver):
    def retrieve_solution(self, model: clingo.Model) -> MisSolution:
        symbols = model.symbols(atoms=True)
        set_is = [s.arguments[0].number for s in symbols if s.name == "independent_set"]
        empty_sol = [0 for i in range(self.problem.number_nodes)]
        for s in set_is:
            empty_sol[s] = 1
        return MisSolution(problem=self.problem, chosen=empty_sol)

    def init_model(self, **kwargs: Any) -> None:
        if self.problem.attribute_aggregate == "size":
            basic_model = """
            {independent_set(V):vertex(V)}.
            independent_set(V) :- vertex(V), not {independent_set(U) : edge(U, V)}.
            :- edge(X,Y), independent_set(X), independent_set(Y).
            #maximize {1, V:independent_set(V)}.
            #show independent_set/1.
            """
            basic_model = """
            % Define the graph
            % Generate a subset of nodes
            { independent_set(X) : vertex(X) }.

            % Constraint: No two adjacent nodes can be in the independent set
            :- independent_set(X), independent_set(Y), edge(X,Y).

            % WIP Symmetry breaking: Choose the node with the smallest index among its neighbors
            %:- independent_set(Y), vertex(X), X < Y, vertex(X,Y), not independent_set(X), not any_smaller_in(Y).
            any_smaller_in(Y) :- independent_set(X), X < Y, edge(X,Y).

            % Heuristic: Prefer nodes with fewer neighbors
            #heuristic independent_set(X) : vertex(X), N = #count{ Y : edge(X,Y) }. [N@1, false]

            % Optimization: Maximize the size of the independent set
            #maximize { 1,X : independent_set(X) }.
            % Output
            #show independent_set/1.
            """
        else:
            basic_model = """
                        {independent_set(V):vertex(V)}.
                        independent_set(V) :- vertex(V), not {independent_set(U) : edge(U, V)}.
                        :- edge(X,Y), independent_set(X), independent_set(Y).
                        #maximize {W,V : independent_set(V), weight(V,W)}.
                        #show independent_set/1.
                        """
        max_models = kwargs.get("max_models", 100)
        self.ctl = clingo.Control(
            ["--warn=no-atom-undefined", f"--models={max_models}", "--opt-mode=optN"]
        )
        self.ctl.add("base", [], basic_model)
        string_data_input = self.build_data_string()
        self.ctl.add("base", [], string_data_input)

    def build_data_string(self):
        s = f"vertex(0..{self.problem.number_nodes-1}).\n"
        s += "% Define the edges in the graph. Modify this according to your graph structure.\n"
        for edge in self.problem.edges:
            s += f"edge({self.problem.nodes_to_index[edge[0]]},{self.problem.nodes_to_index[edge[1]]}).\n"
        if self.problem.attribute_aggregate != "size":
            for i in range(self.problem.number_nodes):
                s += f"weight({i}, {int(self.problem.attr_list[i])}).\n"
        # s += f"independent_set(0).\n"
        # s += f"independent_set(1).\n"

        return s
