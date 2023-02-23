#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import os
import time
from typing import Any, List, Optional

import clingo
from clingo import Symbol

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.knapsack.knapsack_model import (
    KnapsackModel,
    KnapsackSolution,
)
from discrete_optimization.knapsack.solvers.knapsack_solver import SolverKnapsack

cur_folder = os.path.abspath(os.path.dirname(__file__))
logger = logging.getLogger(__name__)


class KnapsackASPSolver(SolverKnapsack):
    def __init__(
        self,
        knapsack_model: KnapsackModel,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
    ):
        SolverKnapsack.__init__(self, knapsack_model=knapsack_model)
        (
            self.aggreg_sol,
            self.aggreg_from_dict_values,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            self.knapsack_model, params_objective_function=params_objective_function
        )
        self.model: Optional[clingo.Control] = None

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
        nb_items = self.knapsack_model.nb_items

        values = [0] + [
            self.knapsack_model.list_items[i].value
            for i in range(self.knapsack_model.nb_items)
        ]
        weights = [0] + [
            self.knapsack_model.list_items[i].weight
            for i in range(self.knapsack_model.nb_items)
        ]
        max_capacity = self.knapsack_model.max_capacity

        items = [
            f"weight({i},{weights[i]}). value({i},{values[i]})."
            for i in range(1, len(weights))
        ]
        logger.debug(
            f"max weight data : #const max_weight={max_capacity}." + " ".join(items)
        )
        return f"#const max_weight={max_capacity}." + " ".join(items)

    def retrieve_solutions(self, list_symbols: List[List[Symbol]]) -> ResultStorage:
        list_solutions_fit = []
        for symbols in list_symbols:
            in_list = [s.arguments[0].number for s in symbols if s.name == "in"]
            list_taken = [
                1 if i in in_list else 0
                for i in range(1, self.knapsack_model.nb_items + 1)
            ]
            solution = KnapsackSolution(
                problem=self.knapsack_model, list_taken=list_taken
            )
            fit = self.aggreg_sol(solution)
            list_solutions_fit += [(solution, fit)]

        return ResultStorage(
            list_solution_fits=list_solutions_fit,
            mode_optim=self.params_objective_function.sense_function,
        )

    def solve(self, **kwargs: Any) -> ResultStorage:
        start_time_grounding = time.perf_counter()
        logger.info(f"Start grounding...")
        self.ctl.ground([("base", [])])
        logger.info(
            f"Grounding programs: ...\n=== Grounding done"
            f" {time.perf_counter() - start_time_grounding} sec ==="
        )

        class CallbackASP:
            def __init__(
                self,
                dump_model_in_folders: bool = False,
            ):
                self.nb_found_models = 0
                self.current_time = time.perf_counter()
                self.model_results = []
                self.symbols_results = []
                self.dump_model_in_folders = dump_model_in_folders

            def on_model(self, m: clingo.Model):
                self.model_results += [m]
                self.symbols_results += [m.symbols(atoms=True)]
                self.nb_found_models += 1
                logger.info(
                    f"=== New Model [{self.nb_found_models}]"
                    f" found after {time.perf_counter()-self.current_time}"
                    f" sec of solving === "
                )
                logger.info(f"=== cost = {m.cost} ===")
                logger.info(f"=== Optimality proven ? {m.optimality_proven} === ")
                if self.dump_model_in_folders:
                    folder_model = os.path.join(
                        cur_folder, f"output-folder/model_{self.nb_found_models}"
                    )
                    create_empty_folder(folder_model)
                    logger.info("Dumping model.txt ...")
                    with open(
                        os.path.join(folder_model, "model.txt"), "w"
                    ) as model_file:
                        model_file.write(str(m))

        timeout_seconds = kwargs.get("timeout_seconds", 100)
        callback = CallbackASP(
            dump_model_in_folders=kwargs.get("dump_model_in_folders", False),
        )

        with self.ctl.solve(on_model=callback.on_model, async_=True) as handle:
            handle.wait(timeout_seconds)
            handle.cancel()
        return self.retrieve_solutions(callback.symbols_results)


def create_empty_folder(folder):
    logger.info(f"Creating empty folder: {folder}")
    if os.path.exists(folder):
        os.removedirs(folder)
    os.makedirs(folder)
