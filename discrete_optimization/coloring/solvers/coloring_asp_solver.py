#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import os
import time
from typing import Any, List, Optional

import clingo
from clingo import Symbol

from discrete_optimization.coloring.coloring_model import (
    ColoringProblem,
    ColoringSolution,
)
from discrete_optimization.coloring.solvers.coloring_solver_with_starting_solution import (
    SolverColoringWithStartingSolution,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

cur_folder = os.path.abspath(os.path.dirname(__file__))
logger = logging.getLogger(__name__)


class ColoringASPSolver(SolverColoringWithStartingSolution):
    def __init__(
        self,
        coloring_model: ColoringProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        """Solver based on Answer Set Programming formulation and clingo solver.

        Args:
            coloring_model (ColoringProblem): coloring problem instance to solve
            params_objective_function (ParamsObjectiveFunction): params of the objective function
            silent_solve_error: if True, raise a warning instead of an error if the underlying instance.solve() crashes
            **args:
        """
        SolverColoringWithStartingSolution.__init__(self, coloring_model=coloring_model)
        (
            self.aggreg_sol,
            self.aggreg_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.coloring_model,
            params_objective_function=params_objective_function,
        )
        self.ctl: Optional[clingo.Control] = None

    def init_model(self, **kwargs: Any) -> None:
        if self.coloring_model.use_subset:
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
            nb_colors = self.coloring_model.count_colors_all_index(solution.colors)
            nb_colors_subset = self.coloring_model.count_colors(solution.colors)
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
        number_of_nodes = self.coloring_model.number_of_nodes
        nodes = f"node(1..{number_of_nodes}).\n"
        edges = ""
        index_nodes_name = self.coloring_model.index_nodes_name
        for e in self.coloring_model.graph.edges:
            edges += f"edge({index_nodes_name[e[0]]+1}, {index_nodes_name[e[1]]+1}). "
        types = ""
        if self.coloring_model.use_subset:
            # TODO : make this work.
            # types += f"max_value({nb_colors_subset}). \n"
            for node in self.coloring_model.subset_nodes:
                if node in self.coloring_model.subset_nodes:
                    types += (
                        f"subset_node({self.coloring_model.index_nodes_name[node]+1}). "
                    )
        constraints = ""
        if self.coloring_model.constraints_coloring:
            constraints = self.constrained_data_input()
        colors = " ".join([f"col(c_{i})." for i in range(nb_colors)])
        full_string_input = (
            nodes + edges + "\n" + colors + "\n" + types + "\n" + constraints + "\n"
        )
        return full_string_input

    def constrained_data_input(self):
        s = ""
        if self.coloring_model.constraints_coloring.color_constraint is not None:
            for node in self.coloring_model.constraints_coloring.color_constraint:
                value = self.coloring_model.constraints_coloring.color_constraint[node]
                index_node = self.coloring_model.index_nodes_name[node]
                s += f"fixed_color({index_node+1}, c_{value}). "
        return s

    def retrieve_solutions(self, list_symbols: List[List[Symbol]]) -> ResultStorage:
        list_solutions_fit = []
        for symbols in list_symbols:
            colors = [
                (s.arguments[0].number, s.arguments[1].name)
                for s in symbols
                if s.name == "color"
            ]
            colors_name = list(set([s[1] for s in colors]))
            colors_to_index = self.compute_clever_colors_map(colors_name)
            colors_vect = [0] * self.coloring_model.number_of_nodes
            for num, color in colors:
                colors_vect[num - 1] = colors_to_index[color]

            solution = ColoringSolution(problem=self.coloring_model, colors=colors_vect)
            fit = self.aggreg_sol(solution)
            list_solutions_fit += [(solution, fit)]
        return ResultStorage(
            list_solution_fits=list_solutions_fit,
            mode_optim=self.params_objective_function.sense_function,
        )

    def compute_clever_colors_map(self, colors_name: List[str]):
        colors_to_protect = set()
        colors_to_index = {}
        if self.coloring_model.has_constraints_coloring:
            colors_to_protect = set(
                [
                    f"c_{x}"
                    for x in self.coloring_model.constraints_coloring.color_constraint.values()
                ]
            )
            colors_to_index = {
                f"c_{x}": x
                for x in self.coloring_model.constraints_coloring.color_constraint.values()
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

    def solve(self, **kwargs: Any) -> ResultStorage:
        start_time_grounding = time.perf_counter()
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
                if m.optimality_proven:
                    return 0
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
