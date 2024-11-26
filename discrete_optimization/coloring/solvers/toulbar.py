#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
from typing import Any, Iterable, Optional

from discrete_optimization.coloring.problem import ColoringProblem, ColoringSolution
from discrete_optimization.coloring.solvers.starting_solution import (
    WithStartingSolutionColoringSolver,
)
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    IntegerHyperparameter,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

try:
    import pytoulbar2
except ImportError:
    toulbar_available = False
else:
    toulbar_available = True

import logging

from discrete_optimization.generic_tools.do_solver import SolverDO, WarmstartMixin
from discrete_optimization.generic_tools.lns_tools import ConstraintHandler
from discrete_optimization.generic_tools.toulbar_tools import (
    ToulbarSolver,
    to_lns_toulbar,
)

logger = logging.getLogger(__name__)


class ToulbarColoringSolver(
    ToulbarSolver, WithStartingSolutionColoringSolver, WarmstartMixin
):
    hyperparameters = (
        ToulbarSolver.hyperparameters
        + WithStartingSolutionColoringSolver.hyperparameters
        + [
            CategoricalHyperparameter(
                name="value_sequence_chain", choices=[True, False], default=False
            ),
            CategoricalHyperparameter(
                name="hard_value_sequence_chain",
                choices=[True, False],
                default=False,
                depends_on=("value_sequence_chain", [True]),
            ),
            IntegerHyperparameter(
                name="tolerance_delta_max",
                low=0,
                high=2,
                default=1,
                depends_on=("value_sequence_chain", [True]),
            ),
            CategoricalHyperparameter(
                name="vns", choices=[None, -4, -3, -2, -1, 0], default=None
            ),
        ]
    )

    def get_range_value(
        self, index_node: int, nb_colors_on_subset: int, nb_colors_all: int
    ):
        node_name = self.problem.nodes_name[index_node]
        if self.problem.has_constraints_coloring:
            nodes = self.problem.constraints_coloring.nodes_fixed()
            if node_name in nodes:
                value = self.problem.constraints_coloring.color_constraint[node_name]
                return range(value, value + 1)
        in_subset = self.problem.is_in_subset_index(index_node)
        ub_i = nb_colors_on_subset if in_subset else nb_colors_all
        return range(ub_i)

    def init_model(self, **kwargs: Any) -> None:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        number_nodes = self.problem.number_of_nodes
        index_nodes_name = self.problem.index_nodes_name
        nb_colors = kwargs.get("nb_colors", None)
        nb_colors_on_subset = kwargs.get("nb_colors_on_subset", nb_colors)
        if nb_colors is None:
            # Run greedy solver to get an upper bound of the bound.
            sol = self.get_starting_solution(**kwargs)
            self.sol = sol
            nb_colors = int(self.problem.evaluate(sol)["nb_colors"])
            nb_colors_all = self.problem.count_colors_all_index(sol.colors)
            nb_colors_on_subset = self.problem.count_colors(sol.colors)
            logger.info(f"{nb_colors_on_subset} colors found by the greedy method ")
        else:
            nb_colors_all = nb_colors
        # we don't have to have a very tight bound.
        model = pytoulbar2.CFN(nb_colors, vns=kwargs["vns"])
        model.AddVariable("max_color", range(nb_colors_on_subset))
        model.AddFunction(["max_color"], range(nb_colors_on_subset))
        range_map = {}
        names_var = []
        for i in range(number_nodes):
            in_subset = self.problem.is_in_subset_index(i)
            range_map[i] = self.get_range_value(
                index_node=i,
                nb_colors_on_subset=nb_colors_on_subset,
                nb_colors_all=nb_colors_all,
            )
            model.AddVariable(f"x_{i}", range_map[i])
            names_var.append(f"x_{i}")
            if in_subset:
                model.AddFunction(
                    [f"x_{i}", "max_color"],
                    [
                        10000 if val1 > val2 else 0
                        for val1 in range_map[i]
                        for val2 in range(nb_colors_on_subset)
                    ],
                )  # encode that x_{i}<=max_color.
                # Problem.AddLinearConstraint([1, -1], [0, i+1], '>=', 0)  # max_color>x_{i} (alternative way ?)
        value_sequence_chain = kwargs["value_sequence_chain"]
        # Warning : don't use this with special "constraints"
        if value_sequence_chain:
            hard_value_sequence_chain = kwargs["hard_value_sequence_chain"]
            tolerance_delta_max = kwargs["tolerance_delta_max"]
            # play with how "fidele" should be the "max_x" variable
            for j in range(number_nodes):
                model.AddVariable(f"max_x_{j}", range(nb_colors_all))
            model.AddFunction([f"max_x_{0}"], [0] + [1000] * (nb_colors_all - 1))
            model.AddFunction([f"x_{0}"], [0] + [1000] * (nb_colors - 1))
            model.AddFunction(
                ["max_color", f"max_x_{number_nodes-1}"],
                [
                    1000 if val1 != val2 else 0
                    for val1 in range(nb_colors)
                    for val2 in range(nb_colors_all)
                ],
            )
            for j in range(1, number_nodes):
                model.AddFunction(
                    [f"max_x_{j-1}", f"max_x_{j}"],
                    [
                        10000 if val1 > val2 or val2 > val1 + tolerance_delta_max else 0
                        for val1 in range(nb_colors)
                        for val2 in range(nb_colors)
                    ],
                )  # Max is increasing but 1 by 1 only.
                model.AddFunction(
                    [f"max_x_{j}", f"x_{j}"],
                    [
                        10000 if val2 > val1 else 0
                        for val1 in range(nb_colors)
                        for val2 in range(nb_colors)
                    ],
                )
                model.AddFunction(
                    [f"max_x_{j - 1}", f"x_{j}"],
                    [
                        10000 if val2 > val1 + tolerance_delta_max else 0
                        for val1 in range(nb_colors)
                        for val2 in range(nb_colors)
                    ],
                )
                if hard_value_sequence_chain:
                    model.AddFunction(
                        [f"max_x_{j}", f"max_x_{j-1}", f"x_{j}"],
                        [
                            0 if val1 == max(val2, val3) else 10000
                            for val1 in range(nb_colors)
                            for val2 in range(nb_colors)
                            for val3 in range(nb_colors)
                        ],
                    )  # x_j <= max_x_{j}
                    model.AddFunction(
                        [f"max_x_{j-1}", f"x_{j}"],
                        [
                            10000 if val2 > val1 + 1 else 0
                            for val1 in range(nb_colors)
                            for val2 in range(nb_colors)
                        ],
                    )
        len_edges = len(self.problem.graph.edges)
        index = 0
        costs_dict = self.default_costs_matrix(
            nb_colors_all=nb_colors_all, nb_colors_on_subset=nb_colors_on_subset
        )
        for e in self.problem.graph.edges:
            if index % 100 == 0:
                logger.info(f"Nb edges introduced {index} / {len_edges}")
            index1 = index_nodes_name[e[0]]
            index2 = index_nodes_name[e[1]]
            costs_i1_i2 = self.get_costs_matrix(
                index1=index1, index2=index2, costs=costs_dict, range_map=range_map
            )
            model.AddFunction([f"x_{index1}", f"x_{index2}"], costs_i1_i2)
            index += 1
        self.model = model
        if hasattr(self, "sol") and kwargs["greedy_start"]:
            self.set_warm_start(self.sol)

    def default_costs_matrix(self, nb_colors_all: int, nb_colors_on_subset: int):
        costs = [
            10000 if val1 == val2 else 0
            for val1 in range(nb_colors_all)
            for val2 in range(nb_colors_all)
        ]
        costs_dict = {}
        if True:
            costs_dict = {
                "out-out": costs,
                "in-out": [
                    10000 if val1 == val2 else 0
                    for val1 in range(nb_colors_on_subset)
                    for val2 in range(nb_colors_all)
                ],
                "out-in": [
                    10000 if val1 == val2 else 0
                    for val1 in range(nb_colors_all)
                    for val2 in range(nb_colors_on_subset)
                ],
                "in-in": [
                    10000 if val1 == val2 else 0
                    for val1 in range(nb_colors_on_subset)
                    for val2 in range(nb_colors_on_subset)
                ],
            }

        return costs_dict

    def get_costs_matrix(
        self,
        index1: int,
        index2: int,
        costs: dict[str, list],
        range_map: dict[int, Any],
    ):
        in_subset_index1 = self.problem.is_in_subset_index(index1)
        in_subset_index2 = self.problem.is_in_subset_index(index2)
        key = "out-out"
        if in_subset_index1 and in_subset_index2:
            key = "in-in"
        if not in_subset_index1 and in_subset_index2:
            key = "out-in"
        if in_subset_index1 and not in_subset_index2:
            key = "in-out"
        if not in_subset_index1 and not in_subset_index2:
            key = "out-out"
        if not self.problem.has_constraints_coloring:
            return costs[key]
        nodes_fixed = self.problem.constraints_coloring.nodes_fixed()
        node_index1 = self.problem.index_to_nodes_name[index1]
        node_index2 = self.problem.index_to_nodes_name[index2]
        if not (node_index1 in nodes_fixed or node_index2 in nodes_fixed):
            return costs[key]
        else:
            return [
                10000 if val1 == val2 else 0
                for val1 in range_map[index1]
                for val2 in range_map[index2]
            ]

    def retrieve_solution(
        self, solution_from_toulbar2: tuple[list, float, int]
    ) -> Solution:
        return ColoringSolution(
            problem=self.problem,
            colors=solution_from_toulbar2[0][1 : 1 + self.problem.number_of_nodes],
        )

    def set_warm_start(self, solution: ColoringSolution) -> None:
        max_color = max(solution.colors)
        self.model.CFN.wcsp.setBestValue(0, max_color)
        for i in range(1, self.problem.number_of_nodes + 1):
            self.model.CFN.wcsp.setBestValue(i, solution.colors[i - 1])


ToulbarColoringSolverForLns = to_lns_toulbar(ToulbarColoringSolver)


class ColoringConstraintHandlerToulbar(ConstraintHandler):
    def remove_constraints_from_previous_iteration(
        self, solver: SolverDO, previous_constraints: Iterable[Any], **kwargs: Any
    ) -> None:
        pass

    def __init__(self, fraction_node: float = 0.3):
        self.fraction_node = fraction_node

    def adding_constraint_from_results_store(
        self,
        solver: ToulbarColoringSolverForLns,
        result_storage: ResultStorage,
        **kwargs: Any,
    ) -> Iterable[Any]:
        best_sol: ColoringSolution = result_storage.get_best_solution_fit()[0]
        max_ = max(best_sol.colors)
        problem: ColoringProblem = solver.problem
        random_indexes = random.sample(
            range(1, problem.number_of_nodes + 1),
            k=int(self.fraction_node * problem.number_of_nodes),
        )
        text = ",".join(
            f"{index}={best_sol.colors[index - 1]}"
            for index in random_indexes
            if best_sol.colors[index - 1] < max_
        )
        text = "," + text
        # circumvent some timeout issue when calling Parse(text). TODO : investigate.
        solver.model.CFN.timer(100)
        try:
            solver.model.Parse(text)
        except Exception as e:
            logger.warning(f"Error raised during parsing certificate : {e}")
        solver.set_warm_start(best_sol)
