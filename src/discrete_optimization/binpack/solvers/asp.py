#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from typing import Any, List

import clingo

from discrete_optimization.binpack.problem import BinPackProblem, BinPackSolution
from discrete_optimization.generic_tools.asp_tools import AspClingoSolver
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)


class AspBinPackingSolver(AspClingoSolver):
    """Solver based on Answer Set Programming formulation and clingo solver for Bin Packing."""

    problem: BinPackProblem
    upper_bound: int

    def init_model(self, **kwargs: Any) -> None:
        # 1. Define the ASP program
        # We assume the number of available bins is at most the number of items (worst case).
        upper_bound = kwargs.get("upper_bound", self.problem.nb_items)
        self.upper_bound = upper_bound
        basic_model = """
        % --- Constants provided via string input ---
        % max_capacity: capacity of a bin
        % nb_items: total number of items
        % nb_bins: number of bins (upper bound)

        % --- Domains ---
        bin(0..nb_bins-1).

        % --- Generator: Put every item into exactly one bin ---
        1 { put(I, B) : bin(B) } 1 :- weight(I, _).

        % --- Constraints ---

        % 1. Capacity Constraint: The sum of weights in a bin must not exceed max_capacity
        :- bin(B), #sum { W, I : put(I, B), weight(I, W) } > max_capacity.

        % 2. Incompatibility Constraint: Two incompatible items cannot be in the same bin
        :- incompatible(I, J), put(I, B), put(J, B).

        % --- Auxiliary predicates for optimization ---

        % A bin is 'used' if at least one item is put in it
        used(B) :- put(_, B).

        % --- Symmetry Breaking (Optional but recommended) ---
        % To avoid exploring permutations of empty bins, we enforce that
        % bin B can only be used if bin B-1 is used (for B > 0).
        :- used(B), not used(B-1), B > 0.

        % --- Optimization ---
        % Minimize the number of used bins
        #minimize { 1, B : used(B) }.

        #show put/2.
        """

        max_models = kwargs.get("max_models", 1)
        # Initialize Clingo Control
        # --opt-mode=optN finds optimal models
        self.ctl = clingo.Control(
            ["--warn=no-atom-undefined", f"--models={max_models}", "--opt-mode=optN"]
        )

        # Add the logic model
        self.ctl.add("base", [], basic_model)

        # Build and add the data facts
        string_data_input = self.build_string_data_input()
        self.ctl.add("base", [], string_data_input)

    def build_string_data_input(self) -> str:
        """
        Converts the BinPackProblem instance data into ASP facts.
        """
        max_capacity = self.problem.capacity_bin
        nb_items = self.problem.nb_items

        # Create weight facts: weight(ItemIndex, WeightValue).
        # We use the index in the list as the ID for ASP to ensure easy mapping back.
        weights_facts = [
            f"weight({i},{self.problem.list_items[i].weight})." for i in range(nb_items)
        ]

        # Create incompatibility facts: incompatible(ItemIndex1, ItemIndex2).
        incompatible_facts = []
        if self.problem.incompatible_items:
            for i, j in self.problem.incompatible_items:
                incompatible_facts.append(f"incompatible({i},{j}).")

        # Constants
        constants = [
            f"#const max_capacity={max_capacity}.",
            f"#const nb_items={nb_items}.",
            f"#const nb_bins={self.upper_bound}.",
        ]

        # Combine all parts
        full_string = (
            " ".join(constants)
            + " "
            + " ".join(weights_facts)
            + " "
            + " ".join(incompatible_facts)
        )

        logger.debug(f"ASP Input Data: {full_string}")
        return full_string

    def retrieve_solution(self, model: clingo.Model) -> BinPackSolution:
        """
        Parses the Clingo model to construct a BinPackSolution.
        """
        # Extract 'put(I, B)' symbols
        logger.info(
            f"Proven optimality : {model.optimality_proven}, current objective function {model.cost[0], model.optimality_proven}"
        )
        symbols = model.symbols(atoms=True)

        # Initialize allocation list with -1 or a default value
        allocation = [-1] * self.problem.nb_items

        for s in symbols:
            if s.name == "put":
                # s.arguments[0] is Item Index, s.arguments[1] is Bin Index
                item_idx = s.arguments[0].number
                bin_idx = s.arguments[1].number

                # Safety check to ensure we stay within bounds
                if 0 <= item_idx < len(allocation):
                    allocation[item_idx] = bin_idx

        return BinPackSolution(problem=self.problem, allocation=allocation)
