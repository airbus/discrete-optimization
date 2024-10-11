#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import re
from typing import Any, List, Optional

import didppy as dp
import numpy as np

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import ResultStorage
from discrete_optimization.generic_tools.dyn_prog_tools import DpSolver
from discrete_optimization.tsp.problem import TspProblem, TspSolution
from discrete_optimization.tsp.solvers import TspSolver
from discrete_optimization.tsp.utils import build_matrice_distance

logger = logging.getLogger(__name__)


class DpTspSolver(TspSolver, DpSolver):
    problem: TspProblem
    hyperparameters = DpSolver.hyperparameters

    def init_model(self, **kwargs: Any) -> None:
        model = dp.Model()
        distance_matrix = build_matrice_distance(
            self.problem.node_count,
            method=self.problem.evaluate_function_indexes,
        )
        closest = np.argsort(distance_matrix, axis=1)
        new_mat = 10 * distance_matrix
        # new_mat = 100000*np.ones(distance_matrix.shape)
        for c in range(distance_matrix.shape[0]):
            new_mat[c, closest[c, :20]] = distance_matrix[c, closest[c, :20]]
            new_mat[c, self.problem.start_index] = distance_matrix[
                c, self.problem.start_index
            ]
            new_mat[c, self.problem.end_index] = distance_matrix[
                c, self.problem.end_index
            ]
        # new_mat = distance_matrix
        c = [
            [int(10 * new_mat[i, j]) for j in range(new_mat.shape[1])]
            for i in range(new_mat.shape[0])
        ]
        distance = model.add_int_table(c)
        customer = model.add_object_type(number=self.problem.node_count)
        start = self.problem.start_index
        end = self.problem.end_index
        unvisited = model.add_set_var(
            object_type=customer,
            target=list([i for i in range(self.problem.node_count) if i != start]),
        )
        location = model.add_element_var(object_type=customer, target=start)
        model.add_base_case([unvisited.is_empty(), location == end])
        for i in range(self.problem.node_count):
            if i not in {end, start}:
                visit = dp.Transition(
                    name=f"{i}",
                    cost=distance[location, i] + dp.IntExpr.state_cost(),
                    effects=[(unvisited, unvisited.remove(i)), (location, i)],
                    preconditions=[unvisited.contains(i)],
                )
                model.add_transition(visit)
            elif end != start and i == end:
                visit = dp.Transition(
                    name=f"{i}",
                    cost=distance[location, i] + dp.IntExpr.state_cost(),
                    effects=[(unvisited, unvisited.remove(i)), (location, i)],
                    preconditions=[unvisited.len() == 1, unvisited.contains(i)],
                )
                model.add_transition(visit)
            else:
                visit = dp.Transition(
                    name=f"{i}",
                    cost=distance[location, i] + dp.IntExpr.state_cost(),
                    effects=[(location, i)],
                    preconditions=[unvisited.is_empty()],
                )
                model.add_transition(visit)
        min_distance_to = model.add_int_table(
            [
                min(c[k][j] for k in range(self.problem.node_count) if j != k)
                for j in range(self.problem.node_count)
            ]
        )
        model.add_dual_bound(
            min_distance_to[unvisited]
            + (location != 0).if_then_else(min_distance_to[0], 0)
        )
        self.model = model

    def retrieve_solution(self, sol: dp.Solution) -> Solution:
        path = []
        for t in sol.transitions:
            name = int(t.name)
            path.append(name)
        logger.info(f"Cost: {sol.cost}")
        tsp_sol = TspSolution(
            problem=self.problem,
            start_index=self.problem.start_index,
            end_index=self.problem.end_index,
            permutation=path[:-1],
        )
        return tsp_sol
