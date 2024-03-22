#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import os
import time
from typing import Any, List, Optional

import clingo
from clingo import Symbol

from discrete_optimization.generic_tools.asp_tools import ASPClingoSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.knapsack.knapsack_model import KnapsackSolution
from discrete_optimization.knapsack.solvers.knapsack_solver import SolverKnapsack

cur_folder = os.path.abspath(os.path.dirname(__file__))
logger = logging.getLogger(__name__)


class KnapsackASPSolver(ASPClingoSolver, SolverKnapsack):
    """Solver based on Answer Set Programming formulation and clingo solver."""

    def init_model(self, **kwargs: Any) -> None:
        basic_model = """
        % knapsack bound
        % choice rule: in/1 is a subset of the set of items.
        {in(I):weight(I,W)} :- weight(I,W).

        % integrity constraint: total weight of items in this subset doesn't exceecd d max_weight.
        :- #sum{W,I: in(I), weight(I,W)} > max_weight.

        % optimization: maximize the values for in/1.
        #maximize {V,I : in(I), value(I,V)}.
        %total_value(X) :- X = #sum{V,I: in(I), value(I,V)}.
        %total_weight(X) :- X = #sum{W,I: in(I), weight(I,W)}.
        #show in/1.
        %#show total_value/1.
        %#show total_weight/1.
        """

        max_models = kwargs.get("max_models", 1)

        self.ctl = clingo.Control(
            ["--warn=no-atom-undefined", f"--models={max_models}", "--opt-mode=optN"]
        )
        self.ctl.add("base", [], basic_model)
        string_data_input = self.build_string_data_input()
        self.ctl.add("base", [], string_data_input)

    def build_string_data_input(self):
        nb_items = self.problem.nb_items

        values = [0] + [
            self.problem.list_items[i].value for i in range(self.problem.nb_items)
        ]
        weights = [0] + [
            self.problem.list_items[i].weight for i in range(self.problem.nb_items)
        ]
        max_capacity = self.problem.max_capacity

        items = [
            f"weight({i},{weights[i]}). value({i},{values[i]})."
            for i in range(1, len(weights))
        ]
        logger.debug(
            f"max weight data : #const max_weight={max_capacity}." + " ".join(items)
        )
        return f"#const max_weight={max_capacity}." + " ".join(items)

    def retrieve_solution(self, model: clingo.Model) -> KnapsackSolution:
        symbols = model.symbols(atoms=True)
        in_list = [s.arguments[0].number for s in symbols if s.name == "in"]
        list_taken = [
            1 if i in in_list else 0 for i in range(1, self.problem.nb_items + 1)
        ]
        return KnapsackSolution(problem=self.problem, list_taken=list_taken)
